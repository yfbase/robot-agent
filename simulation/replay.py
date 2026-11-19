"""Deterministic visual replay of a recorded episode (no LLM, no re-sim).

Reads ``trajectory.npz`` (qpos/ctrl per physics step) from an experiment
episode directory and plays the recorded joint/ball states back through the
same scene, re-rendering both cameras to MP4. Because it replays recorded
states rather than re-integrating, it reproduces the episode exactly,
including the kinematic grasp carry.

Run:
    MUJOCO_GL=egl uv run python examples/tennis_pick/replay.py \
        examples/tennis_pick/experiments/<run>/episode_00
"""

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root (simulation package)
from simulation import CAMERAS, FRAME_HEIGHT, FRAME_WIDTH, VIDEO_FPS, TennisPickSim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    traj = np.load(args.episode_dir / "trajectory.npz")
    out = args.out or args.episode_dir / "replay"
    out.mkdir(parents=True, exist_ok=True)

    sim = TennisPickSim(
        num_balls=int(traj["num_balls"]), seed=int(traj["seed"]), record_video=False
    )
    writers = {
        cam: cv2.VideoWriter(
            str(out / f"{cam}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            VIDEO_FPS,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        for cam in CAMERAS
    }
    qpos = traj["qpos"]
    times = traj["time"]
    next_frame = 0.0
    for i in range(len(qpos)):
        sim.data.qpos[:] = qpos[i]
        mujoco.mj_forward(sim.model, sim.data)
        if times[i] >= next_frame:
            next_frame = times[i] + 1.0 / VIDEO_FPS
            for cam in CAMERAS:
                frame = cv2.cvtColor(sim.capture(cam), cv2.COLOR_RGB2BGR)
                writers[cam].write(frame)
    for w in writers.values():
        w.release()
    print("replay written to", out)


if __name__ == "__main__":
    main()
