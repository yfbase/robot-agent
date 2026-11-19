"""Grid search over gripper torque limit and lateral offset (vertical approach).

The vertical grasp almost worked: the ball stayed between the jaws ~1 s and
then slid out sideways under sustained full-torque squeeze. This sweep tests
whether a lower gripper torque limit (real servos support torque limiting)
and/or a different lateral offset produces a stable hold and a clean drop.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/tune_grasp_grid.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import BOX_CENTER_XY, TennisPickSim
from tune_grasp import ik_solve, save_view


def attempt(force: float, offset: float, seed: int = 7) -> dict:
    sim = TennisPickSim(num_balls=1, seed=seed, record_video=False)
    gid = sim.model.actuator("gripper").id
    sim.model.actuator_forcerange[gid] = [-force, force]
    ball = sim.ball_position(0)
    tag = f"f{force:.1f}_o{offset:.3f}"

    hover = ball + np.array([0.0, 0.0, 0.06])
    sim.set_servo_angles(ik_solve(sim, hover, lateral_offset=offset))
    sim.set_servo_angles(ik_solve(sim, ball - np.array([0.0, 0.0, 0.01]), lateral_offset=offset))
    sim.grasp()
    sim.set_servo_angles(ik_solve(sim, hover, lateral_offset=offset))
    held = sim.ball_position(0)[2] > 0.06
    sim.set_servo_angles(
        ik_solve(sim, np.array([BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.09]), lateral_offset=offset)
    )
    sim.release()
    sim.set_servo_angles(ik_solve(sim, np.array([BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.09])))
    in_box = sim.ball_in_box(0)
    print(f"{tag}: held={held} in_box={in_box} final={sim.ball_position(0).round(3)}")
    if held:
        save_view(sim, f"grid_{tag}")
    return {"force": force, "offset": offset, "held": bool(held), "in_box": bool(in_box)}


def main():
    results = []
    for force in (3.35, 1.5, 0.8):
        for offset in (0.03, 0.035, 0.04):
            results.append(attempt(force, offset))
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
