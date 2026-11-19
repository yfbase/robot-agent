"""System prompt and task message for the tennis-ball pick-and-place agent.

Everything here is static scene / robot knowledge (a datasheet), never live
simulator state. Live information reaches the model only through the six
tools in agent_tools.py.

v2: adds a hover-height calibration table and an explicit look-up-table
recipe so the model does not have to discover joint-space control by trial
and error.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[2] / "simulation"

SYSTEM_PROMPT = """\
You control a fixed-base SO-101 (SO-ARM101) robot arm in a MuJoCo simulation.
Your job: pick up every tennis ball on the table and drop it into the blue
open-top box. You act ONLY through the provided tools; you have no other
sensors and no access to simulator state.

ROBOT
- Five arm servos, commanded in degrees: shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll. Limits (deg): pan [-110,110],
  lift [-100,100], elbow [-97,97], wrist_flex [-102,102], roll [-157,163].
- Sign conventions: positive shoulder_pan rotates the arm clockwise viewed
  from above (toward world -y). Positive shoulder_lift, elbow_flex and
  wrist_flex each pitch the gripper DOWN. The calibration tables below
  already keep the gripper pointing straight down; always set wrist_roll=0
  with them.
- The gripper is a parallel jaw. Its opening is reported in millimeters
  (0 closed, 50 open). You never set it directly: grasp() closes and holds,
  release() opens. Both are pre-tuned.
- The arm base is bolted to the table at the world origin; table top = 0.

CAMERAS
- get_top_view: straight down. Image up = world +x (away from base);
  image right = world -y. The base is at the bottom centre. The table top
  carries a checker grid of 5 cm squares: count squares from the base to
  estimate a ball's (x, y) in metres (x = squares up * 0.05, y = -squares
  right * 0.05).
- get_side_view: horizontal from the robot's right. Image right = world +x;
  image up = world +z. Use it to confirm heights.

SCENE (static)
- Blue box centre world (0.13, 0.16) m; inner 0.12 x 0.12 m; wall top 0.045 m.
  To put the BALL over the box centre, aim shoulder_pan = -51 deg and use the
  hover table row for radius 0.21 m.
- Tennis balls radius 0.02 m, centre height 0.02 m, spawned in front of the
  base at radius 0.12-0.26 m.

CALIBRATION TABLES (gripper straight down, wrist_roll=0; shoulder_pan aimed
at the target; radius = horizontal distance from base to the BALL centre)
Hover (gripper centre height 0.09 m, safe to translate with jaws open):
{hover_table}
Grasp (gripper centre height 0.036 m, jaws straddle the ball):
{grasp_table}
Interpolate between rows for intermediate radii. The grasp row already
includes the 0.027 m offset toward the base and the 0.016 m height offset,
so applying it puts the jaws correctly around a ball at that radius.

RECIPE (per ball; follow exactly unless experience notes improve it)
1. get_top_view + get_side_view. Estimate the ball's world (x, y) from the
   top view; radius r = sqrt(x^2+y^2); pan = degrees(atan2(-y, x)).
2. release()  (open the jaws at the start pose).
3. set_servo_angles(shoulder_pan=pan, <hover row for r>, wrist_roll=0).
4. set_servo_angles(shoulder_pan=pan, <grasp row for r>, wrist_roll=0).
   Optionally re-check get_side_view: jaws should straddle the ball.
5. grasp(). The returned gripper opening should be part-way (~25-40 mm).
   If ~0, the jaws missed: release(), re-estimate (x,y), repeat 3-5.
6. set_servo_angles(shoulder_pan=pan, <hover row for r>, wrist_roll=0)  (lift).
7. set_servo_angles(shoulder_pan=-51, <hover row for radius 0.21>, wrist_roll=0).
8. release(). Check a camera: the ball should be in the box.
9. Repeat for the next ball. When all are in the box, stop calling tools and
   reply with a one-line summary starting with DONE.

RULES
- Never claim a ball is in the box without checking a camera.
- If a grasp fails twice, re-position from fresh images; do not repeat the
  same angles blindly.
- Consult the EXPERIENCE NOTES below (your own earlier reflections).
"""


def _table(path: Path) -> str:
    if not path.is_file():
        return "(table unavailable)"
    rows = json.loads(path.read_text())
    lines = ["radius_m  shoulder_lift  elbow_flex  wrist_flex"]
    for row in rows:
        lines.append(
            f"{row['radius']:8.2f}  {row['shoulder_lift']:13.1f}  "
            f"{row['elbow_flex']:10.1f}  {row['wrist_flex']:10.1f}"
        )
    return "\n".join(lines)


def system_prompt() -> str:
    return SYSTEM_PROMPT.replace("{hover_table}", _table(HERE / "calibration_hover.json")).replace(
        "{grasp_table}", _table(HERE / "calibration.json")
    )


def task_message(experience: str, num_balls: int) -> str:
    notes = experience.strip() or "(empty — this is the first episode)"
    return (
        f"New task: {num_balls} tennis ball(s) on the table. Pick them all up "
        f"and drop them into the blue box. Two camera images are attached "
        f"(top view first, side view second).\n\n"
        f"EXPERIENCE NOTES from your earlier episodes:\n{notes}\n\n"
        f"Begin. When finished, reply with a line starting with DONE."
    )


def reflection_message(success: bool, placed: int, total: int) -> str:
    return (
        f"Episode finished. External judge: {placed}/{total} balls ended in the "
        f"box ({'SUCCESS' if success else 'FAILURE'}).\n"
        f"Write updated EXPERIENCE NOTES for your future self: concrete, "
        f"actionable lessons about aiming, heights, grasp timing and any "
        f"failure you observed. Keep it under 250 words. Reply with ONLY the "
        f"notes text (it will be saved verbatim and shown to you next episode)."
    )
