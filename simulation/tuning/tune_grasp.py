"""Offline scripted pick-and-place to validate the physics (no agent, no LLM).

Uses a small damped-least-squares IK on the gripperframe site to drive the
arm through the SAME public interface the agent tools use (set_servo_angles,
grasp, release). Sweeps the descent height to find where the jaws enclose the
ball, then checks the ball survives lift, transport and release into the box.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/tune_grasp.py
"""

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
from simulation import ARM_SERVOS, BOX_CENTER_XY, TennisPickSim

OUT = Path(__file__).resolve().parent / "renders"


def ik_solve(
    sim: TennisPickSim,
    target: np.ndarray,
    lateral_offset: float = 0.0,
    depth: float = 0.0,
    tilt_deg: float = 0.0,
    iters: int = 300,
) -> dict[str, float]:
    """Position + gripper-orientation IK; returns arm angles in degrees.

    The gripperframe site's X axis is the approach direction (fingertips
    direction); the jaws open along the site Z axis, and the site sits at the
    FIXED jaw tip. ``target`` is the desired BALL center. The site goal is
    derived from it:

    * ``lateral_offset`` shifts against the opening direction so the ball
      sits between the jaws instead of under the fixed jaw;
    * ``depth`` shifts along the approach direction (site stays behind the
      ball center by this much);
    * ``tilt_deg`` tilts the approach away from straight down, toward the
      opening direction. Tilting makes the moving jaw's inner face vertical
      at contact, so the squeeze is horizontal instead of pushing the ball
      into the table.
    """
    model, data = sim.model, sim.data
    sid = model.site("gripperframe").id
    qadr = [model.joint(j).qposadr[0] for j in ARM_SERVOS]
    dofadr = [model.joint(j).dofadr[0] for j in ARM_SERVOS]
    limits = np.array([model.joint(j).range for j in ARM_SERVOS])
    tilt = np.deg2rad(tilt_deg)
    # Analytic 2-link seed: wrist sits ~0.105 m above the site when the
    # gripper points down. Seeding avoids IK local minima at joint limits.
    L1, L2 = 0.1126, 0.1349
    shoulder = np.array([0.039, 0.062])
    pan0 = np.arctan2(-target[1], target[0])
    open_h = np.array([np.cos(pan0), -np.sin(pan0), 0.0])
    goal0 = target - open_h * lateral_offset + np.array([0.0, 0.0, -depth])
    wrist = goal0 + np.array([0.0, 0.0, 0.089])
    dr = np.hypot(wrist[0], wrist[1])
    dz = wrist[2] - shoulder[1]
    dist = np.clip(np.hypot(dr, dz), abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4)
    interior = np.arccos(np.clip((L1**2 + L2**2 - dist**2) / (2 * L1 * L2), -1, 1))
    b = np.pi - interior
    a = np.arctan2(dz, dr) - np.arctan2(L2 * np.sin(interior), L1 + L2 * np.cos(interior))
    g = np.pi / 2 - a - b
    pan = np.arctan2(-target[1], target[0])
    seed = np.clip([pan, a, b, g, 0.0], limits[:, 0], limits[:, 1])
    data.qpos[qadr] = seed
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        rot = data.site_xmat[sid].reshape(3, 3)
        opening_dir = rot[:, 2]
        approach_desired = -np.cos(tilt) * np.array([0.0, 0.0, 1.0]) + np.sin(tilt) * opening_dir
        approach_desired /= np.linalg.norm(approach_desired)
        goal = target - opening_dir * lateral_offset + approach_desired * depth
        err_p = goal - data.site_xpos[sid]
        approach = rot[:, 0]
        err_r = 0.5 * np.cross(approach, approach_desired)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        jac = np.vstack([jacp[:, dofadr], 0.15 * jacr[:, dofadr]])
        err = np.concatenate([err_p, 0.15 * err_r])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), err)
        q = data.qpos[qadr] + np.clip(dq, -0.08, 0.08)
        data.qpos[qadr] = np.clip(q, limits[:, 0], limits[:, 1])
        if np.linalg.norm(err_p) < 1e-4:
            break
    mujoco.mj_forward(model, data)
    return {j: float(np.rad2deg(data.qpos[a])) for j, a in zip(ARM_SERVOS, qadr)}


def save_view(sim, name):
    OUT.mkdir(exist_ok=True)
    for cam in ("top_view", "side_view"):
        img = cv2.cvtColor(sim.capture(cam), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT / f"{name}_{cam}.png"), img)


def attempt(tilt: float, lateral_offset: float, seed: int = 7) -> dict:
    sim = TennisPickSim(num_balls=1, seed=seed, record_video=False)
    ball = sim.ball_position(0)
    tag = f"t{tilt:.0f}_o{lateral_offset:.3f}"
    print(f"\n=== tilt {tilt} deg, lateral offset {lateral_offset} m | ball {ball.round(3)} ===")

    hover = ball + np.array([0.0, 0.0, 0.06])
    sim.set_servo_angles(ik_solve(sim, hover, lateral_offset=lateral_offset, tilt_deg=tilt))
    save_view(sim, f"hover_{tag}")

    sim.set_servo_angles(
        ik_solve(sim, ball, lateral_offset=lateral_offset, depth=0.03, tilt_deg=tilt)
    )
    save_view(sim, f"descend_{tag}")
    print("gripper at descent:", sim.data.site("gripperframe").xpos.round(3))

    sim.grasp()
    save_view(sim, f"grasp_{tag}")
    sim.set_servo_angles(
        ik_solve(sim, hover, lateral_offset=lateral_offset, depth=0.03, tilt_deg=tilt)
    )
    ball_after_lift = sim.ball_position(0)
    held = ball_after_lift[2] > 0.06
    print(f"after lift: ball {ball_after_lift.round(3)} held={held}")
    save_view(sim, f"lift_{tag}")

    over_box = np.array([BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.09])
    sim.set_servo_angles(
        ik_solve(sim, over_box, lateral_offset=lateral_offset, depth=0.03, tilt_deg=tilt)
    )
    save_view(sim, f"over_box_{tag}")
    sim.release()
    sim.set_servo_angles(ik_solve(sim, over_box))
    final = sim.ball_position(0)
    in_box = sim.ball_in_box(0)
    print(f"final: ball {final.round(3)} in_box={in_box}")
    save_view(sim, f"final_{tag}")
    return {"tilt": tilt, "offset": lateral_offset, "held": bool(held), "in_box": bool(in_box)}


def main():
    results = [
        attempt(20, 0.03),
        attempt(25, 0.03),
        attempt(25, 0.04),
        attempt(30, 0.035),
    ]
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
