"""MuJoCo simulation package for the SO-101 tennis-ball pick-and-place task.

Holds the simulation engine, scene, official SO-101 model (+ local
modifications), calibration tables, offline tuning probes, experiment records
and the deterministic replay tool. The agent-facing application (tools,
prompts, runner) lives in ``examples/tennis_pick/`` and imports this package.
"""

from .engine import (
    ALL_SERVOS,
    ARM_SERVOS,
    BALL_RADIUS,
    BOX_CENTER_XY,
    BOX_INNER_HALF,
    BOX_WALL_TOP,
    CAMERAS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GRIPPER_CLOSE_M,
    GRIPPER_KP,
    GRIPPER_OPEN_M,
    READY_POSE_DEG,
    SCENE_PATH,
    SPAWN_RADIUS,
    SPAWN_X,
    SPAWN_Y,
    VIDEO_FPS,
    TennisPickSim,
)

__all__ = [
    "ALL_SERVOS",
    "ARM_SERVOS",
    "BALL_RADIUS",
    "BOX_CENTER_XY",
    "BOX_INNER_HALF",
    "BOX_WALL_TOP",
    "CAMERAS",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "GRIPPER_CLOSE_M",
    "GRIPPER_KP",
    "GRIPPER_OPEN_M",
    "READY_POSE_DEG",
    "SCENE_PATH",
    "SPAWN_RADIUS",
    "SPAWN_X",
    "SPAWN_Y",
    "VIDEO_FPS",
    "TennisPickSim",
]