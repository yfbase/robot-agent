"""The six agent-facing tools for the tennis-ball pick-and-place task.

These are the ONLY capabilities the model gets. They are async so
pydantic-ai runs them on the event-loop (main) thread: MuJoCo EGL contexts
are thread-bound and rendering from worker threads segfaults. They return exactly what a
real robot's cameras and joint encoders would return — no object positions,
no contact forces, no simulator state. Grasp success is decided inside the
simulation (a gripper "knows" it holds something); the model is never told
where the ball is.

Camera images are returned as JPEG ``BinaryContent`` so a vision model sees
them as the tool result, mirroring ``robot_agent.tools.capture_photo``.
"""

import sys
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import BinaryContent

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from simulation import ARM_SERVOS, TennisPickSim


def make_tools(sim: TennisPickSim, image_dir: Path | None = None, log: Callable | None = None):
    """Build the tool whitelist bound to one simulation instance.

    ``image_dir``: where each captured frame is archived (experiment record).
    ``log``: optional callable(name, args, result_summary) for the tool log.
    """
    counter = {"n": 0}

    def record(name: str, args: dict, summary: str) -> None:
        if log is not None:
            log(name, args, summary)

    def camera(view: str) -> BinaryContent:
        counter["n"] += 1
        data = sim.capture_jpeg(view)
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{counter['n']:04d}_{view}.jpg").write_bytes(data)
        record(f"get_{view}", {}, f"image {len(data)} bytes")
        return BinaryContent(data=data, media_type="image/jpeg")

    async def get_top_view() -> BinaryContent:
        """Overhead camera looking straight down at the table.

        Image up = away from the robot base; image right = the robot's left.
        Use it to locate the tennis balls and the blue box relative to the
        gripper, and to aim shoulder_pan.
        """
        return camera("top_view")

    async def get_side_view() -> BinaryContent:
        """Side camera looking horizontally across the table.

        Image right = away from the robot base; image up = world up.
        Use it to judge heights: gripper above ball, ball above box wall.
        """
        return camera("side_view")

    async def get_servo_angles() -> dict:
        """Read the current servo angles.

        Returns degrees for shoulder_pan, shoulder_lift, elbow_flex,
        wrist_flex, wrist_roll, and the gripper opening in millimeters
        (0 = fully closed, 50 = fully open).
        """
        angles = sim.servo_angles_deg()
        record("get_servo_angles", {}, str(angles))
        return angles

    async def set_servo_angles(
        shoulder_pan: float | None = None,
        shoulder_lift: float | None = None,
        elbow_flex: float | None = None,
        wrist_flex: float | None = None,
        wrist_roll: float | None = None,
    ) -> dict:
        """Command arm servo angles in degrees and wait until motion settles.

        Omitted servos keep their current command. Values are clamped to the
        joint limits. Returns the achieved angles (degrees) plus the gripper
        opening in millimeters. The gripper itself is NOT controlled here —
        use grasp() / release().

        Sign conventions: positive shoulder_pan turns the arm clockwise seen
        from above (toward the robot's right); positive shoulder_lift,
        elbow_flex and wrist_flex all pitch the gripper downward. Keeping
        shoulder_lift + elbow_flex + wrist_flex near +90 degrees points the
        gripper straight down.
        """
        args = {
            k: v
            for k, v in {
                "shoulder_pan": shoulder_pan,
                "shoulder_lift": shoulder_lift,
                "elbow_flex": elbow_flex,
                "wrist_flex": wrist_flex,
                "wrist_roll": wrist_roll,
            }.items()
            if v is not None
        }
        for name in args:
            if name not in ARM_SERVOS:
                raise ValueError(f"unknown servo {name}")
        achieved = sim.set_servo_angles(args)
        record("set_servo_angles", args, str(achieved))
        return achieved

    async def grasp() -> dict:
        """Close the gripper (pre-tuned setpoint) and hold.

        Call it only when the open jaws straddle a ball at ball height.
        Returns the achieved servo angles; the gripper opening tells you how
        far the jaws closed (a ball between them stops the jaws part-way).
        """
        achieved = sim.grasp()
        record("grasp", {}, str(achieved))
        return achieved

    async def release() -> dict:
        """Open the gripper (pre-tuned setpoint), dropping whatever it holds.

        Call it above the blue box to deposit a ball. Returns achieved angles.
        """
        achieved = sim.release()
        record("release", {}, str(achieved))
        return achieved

    return [get_top_view, get_side_view, get_servo_angles, set_servo_angles, grasp, release]
