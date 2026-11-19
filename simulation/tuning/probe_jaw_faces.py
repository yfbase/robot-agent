"""Measure jaw finger geometry in the gripper body frame vs gripper angle.

In the gripper body frame the approach direction is -Z and the jaws open
along +X (the gripperframe site sits at (-0.008, 0, -0.098), i.e. at the
fixed jaw tip). Prints, for each gripper angle, the fingertip positions and
the inner-face extents so we can compute exactly where a ball must sit.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/probe_jaw_faces.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_gripper import mesh_world_verts
from simulation import TennisPickSim


def in_gripper_frame(sim, verts):
    xpos = sim.data.body("gripper").xpos
    xmat = sim.data.body("gripper").xmat.reshape(3, 3)
    return (verts - xpos) @ xmat


def main():
    sim = TennisPickSim(num_balls=1, seed=7, record_video=False)
    model, data = sim.model, sim.data
    adr = model.joint("gripper").qposadr[0]
    for deg in (-10, 0, 10, 20, 30, 45, 60, 95):
        data.qpos[adr] = np.deg2rad(deg)
        mujoco.mj_forward(model, data)
        fixed = in_gripper_frame(
            sim, mesh_world_verts(model, data, "gripper", "wrist_roll_follower_so101_v1")
        )
        moving = in_gripper_frame(
            sim, mesh_world_verts(model, data, "moving_jaw_so101_v1", "moving_jaw_so101_v1")
        )
        # finger region = verts far along approach (z < -0.06)
        ff = fixed[fixed[:, 2] < -0.06]
        mf = moving[moving[:, 2] < -0.06]
        print(f"gripper={deg:+4d} deg")
        if len(ff):
            print(
                f"  fixed finger: x[{ff[:, 0].min():+.4f},{ff[:, 0].max():+.4f}] "
                f"y[{ff[:, 1].min():+.4f},{ff[:, 1].max():+.4f}] z[{ff[:, 2].min():+.4f},{ff[:, 2].max():+.4f}] n={len(ff)}"
            )
        if len(mf):
            print(
                f"  moving finger: x[{mf[:, 0].min():+.4f},{mf[:, 0].max():+.4f}] "
                f"y[{mf[:, 1].min():+.4f},{mf[:, 1].max():+.4f}] z[{mf[:, 2].min():+.4f},{mf[:, 2].max():+.4f}] n={len(mf)}"
            )
        else:
            print(
                f"  moving finger: (none beyond z=-0.06) full bbox x[{moving[:, 0].min():+.3f},{moving[:, 0].max():+.3f}] z[{moving[:, 2].min():+.3f},{moving[:, 2].max():+.3f}]"
            )


if __name__ == "__main__":
    main()
