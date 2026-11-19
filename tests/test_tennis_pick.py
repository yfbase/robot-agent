"""Offline regression tests for the tennis-ball pick-and-place example.

No API key needed: exercises the simulation physics (scripted reference
policy) and the agent tool whitelist. Run with:
    MUJOCO_GL=egl uv run pytest tests/test_tennis_pick.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tennis_pick"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation" / "tuning"))
sys.path.insert(0, str(EXAMPLE))

from agent_tools import make_tools
from run_experiment import scripted_episode
from simulation import TennisPickSim


def test_scripted_pick_and_place_succeeds():
    for seed in (7, 11):
        sim = TennisPickSim(num_balls=1, seed=seed, record_video=False)
        scripted_episode(sim)
        assert sim.ball_in_box(0), f"seed {seed}: ball not in box"


def test_tool_whitelist_is_exactly_six():
    sim = TennisPickSim(num_balls=1, seed=7, record_video=False)
    tools = make_tools(sim)
    names = {t.__name__ for t in tools}
    assert names == {
        "get_top_view",
        "get_side_view",
        "get_servo_angles",
        "set_servo_angles",
        "grasp",
        "release",
    }


def test_cameras_return_jpeg_and_angles_have_no_ball_position():
    import asyncio

    sim = TennisPickSim(num_balls=1, seed=7, record_video=False)
    tools = {t.__name__: t for t in make_tools(sim)}
    img = asyncio.run(tools["get_top_view"]())
    assert img.media_type == "image/jpeg" and img.data[:3] == b"\xff\xd8\xff"
    angles = asyncio.run(tools["get_servo_angles"]())
    assert set(angles) == {
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    }
