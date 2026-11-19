"""Horizontal-approach grasp test: gripper points sideways at the ball,
jaws open vertically (one finger above, one below).

Why: with a vertical (top-down) approach the two slanted finger faces form a
wedge that closes DOWNWARD, extruding the ball into the table. With a
horizontal approach the same wedge extrudes horizontally, which contact
friction resists with a large self-locking margin, and gravity seats the ball
on the lower finger.

Run:  MUJOCO_GL=egl uv run python examples/tennis_pick/tuning/tune_grasp_horizontal.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (simulation package)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
from simulation import ARM_SERVOS, BOX_CENTER_XY, TennisPickSim
from tune_grasp import save_view

# Ball center relative to the gripperframe site, in gripper-frame axes
# (measured in probe_jaw_faces.py): +0.030 along the opening axis (site z),
# -0.018 along the approach axis (site x). With jaws opening vertically
# downward-opening (site z = -world z), the site sits 0.030 ABOVE the ball
# center and 0.018 further along the approach.
OPEN_OFFSET = 0.030
DEPTH = 0.018


def ik_horizontal(sim: TennisPickSim, ball_target: np.ndarray, iters: int = 400):
    """Place the site for a horizontal side grasp on ``ball_target``.

    Orientation: approach horizontal pointing radially outward (yaw = pan),
    opening axis 20 deg off straight down (wrist_roll ~160 deg, the joint
    limit is 163 deg so straight down is unreachable). Fixed finger above
    the ball, moving finger below.
    """
    model, data = sim.model, sim.data
    sid = model.site("gripperframe").id
    qadr = [model.joint(j).qposadr[0] for j in ARM_SERVOS]
    dofadr = [model.joint(j).dofadr[0] for j in ARM_SERVOS]
    limits = np.array([model.joint(j).range for j in ARM_SERVOS])
    pan = np.arctan2(-ball_target[1], ball_target[0])
    approach_desired = np.array([np.cos(pan), -np.sin(pan), 0.0])
    opening_desired = np.array([0.342 * np.sin(pan), -0.342 * np.cos(pan), -0.940])
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        rot = data.site_xmat[sid].reshape(3, 3)
        approach, opening = rot[:, 0], rot[:, 2]
        goal = ball_target - opening_desired * OPEN_OFFSET + approach_desired * DEPTH
        err_p = goal - data.site_xpos[sid]
        err_r = 0.5 * (np.cross(approach, approach_desired) + np.cross(opening, opening_desired))
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        jac = np.vstack([jacp[:, dofadr], 0.3 * jacr[:, dofadr]])
        err = np.concatenate([err_p, 0.3 * err_r])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        q = data.qpos[qadr] + np.clip(dq, -0.08, 0.08)
        data.qpos[qadr] = np.clip(q, limits[:, 0], limits[:, 1])
        if np.linalg.norm(err_p) < 1e-4 and np.linalg.norm(err_r) < 1e-3:
            break
    mujoco.mj_forward(model, data)
    return {j: float(np.rad2deg(data.qpos[a])) for j, a in zip(ARM_SERVOS, qadr)}


def attempt(seed: int, force: float = 3.35, tag: str = "") -> dict:
    sim = TennisPickSim(num_balls=1, seed=seed, record_video=False)
    sim.model.actuator_forcerange[sim.model.actuator("gripper").id] = [-force, force]
    ball = sim.ball_position(0)
    sim.grasp()  # closed jaws while swinging in
    sim.set_servo_angles(ik_horizontal(sim, ball))
    sim.release()  # open once in position above/beside the ball
    ball = sim.ball_position(0)
    sim.set_servo_angles(ik_horizontal(sim, ball))
    save_view(sim, f"hz_descend_{tag}{seed}")
    sim.grasp()
    save_view(sim, f"hz_grasp_{tag}{seed}")
    lift = ball + np.array([0.0, 0.0, 0.06])
    sim.set_servo_angles(ik_horizontal(sim, lift))
    held = sim.ball_position(0)[2] > 0.06
    save_view(sim, f"hz_lift_{tag}{seed}")
    over_box = np.array([BOX_CENTER_XY[0], BOX_CENTER_XY[1], 0.09])
    sim.set_servo_angles(ik_horizontal(sim, over_box))
    sim.release()
    sim.set_servo_angles(ik_horizontal(sim, over_box + np.array([0.0, 0.0, 0.03])))
    in_box = sim.ball_in_box(0)
    print(
        f"seed={seed} force={force}: held={held} in_box={in_box} final={sim.ball_position(0).round(3)}"
    )
    save_view(sim, f"hz_final_{tag}{seed}")
    return {"seed": seed, "force": force, "held": bool(held), "in_box": bool(in_box)}


def main():
    results = [attempt(seed) for seed in (7, 11, 3, 21)]
    print("\n=== summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
