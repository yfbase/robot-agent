"""Tools the model can call: camera capture and real-arm control.

The camera tool returns the photo as ``BinaryContent``; pydantic-ai sends it
back to the model as the tool result, so a vision-capable model can see the
image. Requires a vision model (e.g. deepseek-v4-flash) and opencv-python.

``make_arm_tools(arm)`` binds a set of arm-control tools to one
``simple_arm.SimpleArm`` instance (real SO101 hardware over the STS3215 serial
bus). Requires pyserial and construct.
"""

import sys

import cv2
from pydantic_ai import BinaryContent


def capture_photo(camera_index: int = 0) -> BinaryContent | str:
    """Take a photo with a local camera and return the image.

    Args:
        camera_index: Which camera to use; 0 is the default camera.
    """
    # DirectShow opens much faster than the default backend on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    try:
        if not cap.isOpened():
            return f"Camera {camera_index} could not be opened. Is it connected and free?"
        # Skip a few warm-up frames so auto-exposure settles.
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            return f"Camera {camera_index} opened but failed to capture a frame."
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return "Failed to encode the photo as JPEG."
        return BinaryContent(data=buffer.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()


def make_arm_tools(arm):
    """Build the tool list bound to one real arm (a ``simple_arm.SimpleArm``).

    Mirrors the tennis-simulation tool surface (examples/tennis_pick/agent_tools.py)
    so prompts transfer between simulation and hardware. Two differences: the
    gripper opening is a percentage (0 closed .. 100 open), not millimetres, and
    there is no camera here — pass ``capture_photo`` separately if needed.

    The tools are plain sync functions: pydantic-ai runs them in worker threads,
    and the serial bus (unlike MuJoCo's rendering context) has no thread
    affinity. Bus failures are returned as error strings so the model can react
    instead of the whole run crashing.
    """
    from simple_arm import Joint

    arm_joints = [j for j in Joint if j is not Joint.GRIPPER]

    def read_angles() -> dict:
        rows = arm.read_all()
        angles = {j.name.lower(): round(rows[j]["deg"], 1) for j in arm_joints}
        angles["gripper_opening_percent"] = round(rows[Joint.GRIPPER]["gripper_pct"], 1)
        return angles

    def get_servo_angles() -> dict | str:
        """Read the current servo angles of the real arm.

        Returns degrees for shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
        wrist_roll, plus gripper_opening_percent (0 = fully closed, 100 = fully
        open). Angle convention: 0 deg = midpoint of that servo's stored travel;
        positive shoulder_lift, elbow_flex and wrist_flex pitch the gripper down.
        """
        try:
            return read_angles()
        except (TimeoutError, RuntimeError, OSError) as exc:
            return f"ERROR reading servo angles: {exc}"

    def set_servo_angles(
        shoulder_pan: float | None = None,
        shoulder_lift: float | None = None,
        elbow_flex: float | None = None,
        wrist_flex: float | None = None,
        wrist_roll: float | None = None,
    ) -> dict | str:
        """Command arm servo angles in degrees and wait until motion settles.

        Omitted servos hold their current position. Values beyond the servo's
        stored travel are truncated by the servo firmware. Returns the achieved
        angles, the gripper opening percent, and a 'settled' flag (False = the
        joints had not reached their targets before the timeout). The gripper
        itself is NOT controlled here — use set_gripper.

        Sign conventions: positive shoulder_pan turns the arm clockwise seen
        from above; positive shoulder_lift, elbow_flex and wrist_flex each pitch
        the gripper DOWN. Keeping shoulder_lift + elbow_flex + wrist_flex near
        +90 degrees points the gripper straight down. Raise the arm before
        changing shoulder_pan, so it never sweeps across the table at low height.
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
        if not args:
            return "ERROR: no servo angles given; specify at least one joint."
        try:
            arm.enable()  # torque on; presets current pose as target, so nothing jumps
            for name, deg in args.items():
                arm.move(Joint[name.upper()], deg)
            settled = arm.wait(timeout=8.0)["settled"]
            result = read_angles()
            result["settled"] = settled
            return result
        except (TimeoutError, RuntimeError, OSError) as exc:
            return f"ERROR commanding servo angles: {exc}"

    def set_gripper(opening_percent: float) -> dict | str:
        """Open or close the gripper: 0 = fully closed, 100 = fully open.

        Values outside 0..100 are clamped. Waits until motion settles and
        returns the achieved opening percent plus a 'settled' flag. When
        closing, settled=False usually means the jaws stalled on an object
        before reaching the target — the achieved opening then indicates
        something is held.
        """
        percent = max(0.0, min(100.0, float(opening_percent)))
        try:
            arm.enable(Joint.GRIPPER)
            arm.gripper(percent)
            settled = arm.wait(Joint.GRIPPER, timeout=5.0)["settled"]
            row = arm.read_one(Joint.GRIPPER)
            return {
                "gripper_opening_percent": round(row["gripper_pct"], 1),
                "settled": settled,
            }
        except (TimeoutError, RuntimeError, OSError) as exc:
            return f"ERROR commanding gripper: {exc}"

    def reset_arm() -> dict | str:
        """Reset the arm: open the gripper and raise the arm to its high pose.

        Lifts first, then turns the base, so the arm never sweeps across the
        table at low height. Returns the final angles, the gripper opening
        percent, and a 'settled' flag (False = timed out before reaching the
        pose).
        """
        try:
            settled = arm.reset()["settled"]
            result = read_angles()
            result["settled"] = settled
            return result
        except (TimeoutError, RuntimeError, OSError) as exc:
            return f"ERROR resetting arm: {exc}"

    return [get_servo_angles, set_servo_angles, set_gripper, reset_arm]
