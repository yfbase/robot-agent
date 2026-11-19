"""Joint-space pose search for the top-down parallel-jaw grasp.

The agent only has joint-angle tools (no IK), so the offline validation uses
the same interface: search (shoulder_lift, elbow_flex, wrist_flex) on a grid
with shoulder_pan aimed at the ball and wrist_roll = 0, keep poses whose
gripperframe site lands at the grasp placement (approach straight down,
opening radial, site a fixed offset from the ball center), and command the
best one through set_servo_angles.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/tune_grasp_search.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import ARM_SERVOS, BOX_CENTER_XY, TennisPickSim
from tune_grasp import save_view

# Grasp placement relative to the ball center (meters):
#   site sits LATERAL against the opening direction (fixed jaw beside ball)
#   and ABOVE the ball center by HEIGHT (pads at ball equator).
LATERAL = 0.027
HEIGHT = 0.016


def search_pose(sim: TennisPickSim, goal: np.ndarray, grid: float = 6.0):
    """Find joint angles placing the gripperframe site at ``goal`` (meters),
    gripper pointing straight down, jaws opening radially.

    Coarse 3-D grid over (lift, elbow, wrist_flex) with pan aimed at the
    goal, then two local refinement passes that also adjust pan (the wrist
    chain adds a small yaw offset, so pan needs a few degrees of compensation).
    """
    model, data = sim.model, sim.data
    sid = model.site("gripperframe").id
    qadr = [model.joint(j).qposadr[0] for j in ARM_SERVOS]
    pan = float(np.rad2deg(np.arctan2(-goal[1], goal[0])))
    best = None
    saved = data.qpos.copy()

    def evaluate(cand):
        nonlocal best
        data.qpos[qadr] = np.deg2rad(cand)
        mujoco.mj_forward(model, data)
        rot = data.site_xmat[sid].reshape(3, 3)
        if rot[2, 0] > -0.98 or abs(rot[2, 2]) > 0.1:
            return
        err = np.linalg.norm(data.site_xpos[sid] - goal)
        if best is None or err < best[0]:
            best = (err, list(cand))

    for a in np.arange(-80, 81, grid):
        for b in np.arange(-95, 96, grid):
            for g in np.arange(-102, 103, grid):
                evaluate([pan, a, b, g, 0.0])
    if best is None:
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        raise RuntimeError("no vertical-gripper pose found near target")
    for span, step in ((6.0, 2.0), (2.0, 0.5)):
        center = np.array(best[1])
        for dp in np.arange(-span, span + 0.1, step):
            for da in np.arange(-span, span + 0.1, step):
                for db in np.arange(-span, span + 0.1, step):
                    for dg in np.arange(-span, span + 0.1, step):
                        evaluate(center + np.array([dp, da, db, dg, 0.0]))
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return {j: float(v) for j, v in zip(ARM_SERVOS, best[1])}, best[0]


def attempt(seed: int, tag: str = "") -> dict:
    sim = TennisPickSim(num_balls=1, seed=seed, record_video=False)
    ball = sim.ball_position(0)
    pan = np.arctan2(-ball[1], ball[0])
    radial = np.array([np.cos(pan), -np.sin(pan), 0.0])
    grasp_site = ball - radial * LATERAL + np.array([0.0, 0.0, HEIGHT])
    sim.grasp()  # closed jaws for the swing
    up, _ = search_pose(sim, np.array([0.144, 0.0, 0.095]))
    sim.set_servo_angles(up)
    hover, _ = search_pose(sim, grasp_site + np.array([0.0, 0.0, 0.06]))
    sim.set_servo_angles(hover)
    sim.release()
    for dz in (0.03, 0.015, 0.007, 0.0):
        pose, err = search_pose(sim, grasp_site + np.array([0.0, 0.0, dz]))
        sim.set_servo_angles(pose)
    pre = np.linalg.norm(sim.ball_position(0)[:2] - ball[:2])
    site = sim.data.site("gripperframe").xpos.copy()
    sim.grasp()
    grip = sim.servo_angles_deg()["gripper"]
    save_view(sim, f"search_grasp_{tag}{seed}")
    lift, _ = search_pose(sim, grasp_site + np.array([0.0, 0.0, 0.06]))
    sim.set_servo_angles(lift)
    held = sim.ball_position(0)[2] > 0.06
    save_view(sim, f"search_lift_{tag}{seed}")
    box_pan = np.arctan2(-BOX_CENTER_XY[1], BOX_CENTER_XY[0])
    box_radial = np.array([np.cos(box_pan), -np.sin(box_pan), 0.0])
    box_site = np.array([BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.09]) - box_radial * LATERAL
    over, _ = search_pose(sim, box_site)
    sim.set_servo_angles(over)
    sim.release()
    retract, _ = search_pose(sim, box_site + np.array([0.0, 0.0, 0.01]))
    sim.set_servo_angles(retract)
    in_box = sim.ball_in_box(0)
    print(
        f"seed={seed}: place_err={err:.4f} pre={pre:.3f} site={site.round(3)} "
        f"grip_mm={grip} held={held} in_box={in_box}"
    )
    return {"seed": seed, "held": bool(held), "in_box": bool(in_box)}


def main():
    results = [attempt(s) for s in (7, 11, 3, 21, 5, 42)]
    print("SUCCESS:", sum(r["in_box"] for r in results), "/", len(results))


if __name__ == "__main__":
    main()
