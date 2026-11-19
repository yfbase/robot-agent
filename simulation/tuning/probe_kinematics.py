"""Offline kinematics probe for the SO-101 model (no agent involved).

Prints the gripper-frame site position for a few joint poses so we can
document joint sign conventions, and measures the jaw opening versus the
gripper servo angle so we can pick open/close setpoints.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/probe_kinematics.py
"""

from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "so101_model"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    return model, data


def set_deg(model, data, **deg):
    for name, value in deg.items():
        data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(value)
    mujoco.mj_forward(model, data)


def gripper_pos(model, data):
    return data.site("gripperframe").xpos.copy()


def jaw_tip_positions(model, data):
    # The moving jaw body and the fixed gripper body: use geom bounding info.
    # Approximate jaw tips by the lowest-z points of the jaw collision meshes
    # in world frame; instead we report the moving jaw body origin distance.
    jaw = data.body("moving_jaw_so101_v1").xpos.copy()
    grip = data.body("gripper").xpos.copy()
    return jaw, grip


def main():
    model, data = load()
    print("=== zero pose ===")
    mujoco.mj_forward(model, data)
    print("gripperframe site:", gripper_pos(model, data))
    for j in ARM_JOINTS + ["gripper"]:
        print(f"  joint {j}: range deg = {np.rad2deg(model.joint(j).range)}")

    print("\n=== single-joint sweeps (gripperframe position, meters) ===")
    for j in ARM_JOINTS:
        for angle in (-60, -30, 0, 30, 60):
            set_deg(model, data, **{j: angle})
            p = gripper_pos(model, data)
            print(f"  {j}={angle:+4d} deg -> gripper at ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        set_deg(model, data, **{j: 0})

    print("\n=== candidate 'ready' poses ===")
    poses = {
        "folded_up": {
            "shoulder_pan": 0,
            "shoulder_lift": -60,
            "elbow_flex": 90,
            "wrist_flex": -30,
            "wrist_roll": 0,
        },
        "forward_mid": {
            "shoulder_pan": 0,
            "shoulder_lift": 30,
            "elbow_flex": -60,
            "wrist_flex": 30,
            "wrist_roll": 0,
        },
        "forward_low": {
            "shoulder_pan": 0,
            "shoulder_lift": 60,
            "elbow_flex": -90,
            "wrist_flex": 30,
            "wrist_roll": 0,
        },
    }
    for name, pose in poses.items():
        set_deg(model, data, **pose)
        p = gripper_pos(model, data)
        mat = data.site("gripperframe").xmat.reshape(3, 3)
        print(f"  {name}: gripper at ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
        print(
            f"    site axes (world): x={mat[:, 0].round(3)} y={mat[:, 1].round(3)} z={mat[:, 2].round(3)}"
        )

    print("\n=== gripper angle sweep: moving-jaw body origin relative to gripper body ===")
    for angle in (-10, 0, 20, 40, 60, 80, 100):
        set_deg(model, data, gripper=angle)
        jaw, grip = jaw_tip_positions(model, data)
        print(
            f"  gripper={angle:+4d} deg -> jaw origin {jaw.round(4)}, gripper origin {grip.round(4)}"
        )


if __name__ == "__main__":
    main()
