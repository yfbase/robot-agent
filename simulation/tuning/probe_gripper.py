"""Measure the gripperframe site orientation and jaw-tip geometry precisely.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/probe_gripper.py
"""

import sys
from pathlib import Path

import cv2
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
from simulation import TennisPickSim

OUT = Path(__file__).resolve().parent / "renders"


def mesh_world_verts(model, data, body_name, mesh_name):
    # Arm geoms are unnamed; find the collision geom by body + mesh name.
    body_id = model.body(body_name).id
    gid = None
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body_id and model.geom_contype[g] > 0:
            if model.mesh(model.geom_dataid[g]).name == mesh_name:
                gid = g
                break
    if gid is None:
        raise ValueError(f"no collision geom with mesh {mesh_name} on body {body_name}")
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[start : start + count].reshape(-1, 3)
    xpos = data.geom_xpos[gid]
    xmat = data.geom_xmat[gid].reshape(3, 3)
    return xpos + verts @ xmat.T


def main():
    sim = TennisPickSim(num_balls=1, seed=7, record_video=False)
    model, data = sim.model, sim.data
    sid = model.site("gripperframe").id
    print("ready pose:")
    print("  site pos:", data.site_xpos[sid].round(4))
    print("  site xmat (columns = site x,y,z axes in world):")
    print(data.site_xmat[sid].reshape(3, 3).round(3))

    jaw = mesh_world_verts(model, data, "moving_jaw_so101_v1", "moving_jaw_so101_v1")
    fixed = mesh_world_verts(model, data, "gripper", "wrist_roll_follower_so101_v1")
    print(
        "  moving jaw verts bbox min:", jaw.min(axis=0).round(4), "max:", jaw.max(axis=0).round(4)
    )
    print(
        "  fixed part verts bbox min:",
        fixed.min(axis=0).round(4),
        "max:",
        fixed.max(axis=0).round(4),
    )

    # Close-up render with sites visible (site group 3).
    option = mujoco.MjvOption()
    option.sitegroup[3] = 1
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = data.site_xpos[sid]
    cam.distance = 0.35
    cam.azimuth = 90
    cam.elevation = -10
    renderer.update_scene(data, camera=cam, scene_option=option)
    img = renderer.render()
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "gripper_closeup.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("wrote gripper_closeup.png (site shown as small frame)")


if __name__ == "__main__":
    main()
