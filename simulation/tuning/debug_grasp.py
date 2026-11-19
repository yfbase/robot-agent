"""Slow-motion grasp debug: log ball motion and contacts while jaws close.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/debug_grasp.py
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulation import TennisPickSim
from tune_grasp import ik_solve

OUT = Path(__file__).resolve().parent / "renders"


def geom_name(model, gid):
    name = model.geom(gid).name
    if name:
        return name
    return f"body:{model.body(model.geom_bodyid[gid]).name}"


def main():
    sim = TennisPickSim(num_balls=1, seed=7, record_video=False)
    model, data = sim.model, sim.data
    ball = sim.ball_position(0)
    offset = 0.035
    sim.set_servo_angles(ik_solve(sim, np.array([ball[0], ball[1], 0.09]), lateral_offset=offset))
    sim.set_servo_angles(
        ik_solve(sim, np.array([ball[0], ball[1], ball[2] - 0.01]), lateral_offset=offset)
    )
    print("descended. site:", data.site("gripperframe").xpos.round(4), "ball:", ball.round(4))

    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = ball
    cam.distance = 0.28
    cam.azimuth = 90
    cam.elevation = -8
    OUT.mkdir(exist_ok=True)

    data.ctrl[model.actuator("gripper").id] = np.deg2rad(-10.0)
    t0 = data.time
    next_log = 0.0
    while data.time - t0 < 1.5:
        mujoco.mj_step(model, data)
        if data.time - t0 >= next_log:
            next_log += 0.1
            contacts = []
            for c in range(data.ncon):
                con = data.contact[c]
                g1, g2 = geom_name(model, con.geom1), geom_name(model, con.geom2)
                if "tennis" in g1 or "tennis" in g2:
                    contacts.append(f"{g1}<->{g2}")
            print(
                f"t={data.time - t0:.2f} ball={sim.ball_position(0).round(4)} "
                f"gripper={np.rad2deg(data.qpos[model.joint('gripper').qposadr[0]]):.1f} "
                f"contacts={contacts}"
            )
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            cv2.imwrite(
                str(OUT / f"debug_close_{next_log:.1f}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )


if __name__ == "__main__":
    main()
