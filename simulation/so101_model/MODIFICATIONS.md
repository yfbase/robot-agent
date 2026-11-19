# Local modifications to the official SO-101 model

Source: TheRobotStudio/SO-ARM100, `Simulation/SO101` (see `SOURCE.json` for the
exact commit). The arm link kinematics, masses and visual meshes are the
official ones. The changes below are simulation-level abstractions needed to
make sphere pick-and-place tractable and stable; each is called out so a
reader can judge fidelity.

## In `so101_new_calib.xml`

1. **Gripper joint: hinge -> parallel slide.** The official model drives the
   moving jaw with a single revolute joint whose finger faces are slanted.
   Closing on a sphere then forms a downward-closing wedge that extrudes the
   ball into the table, and the open jaw sweeps through the ball during
   descent. The real SO-101 linkage approximates parallel translation over
   its working range, so we model the gripper as a prismatic (slide) joint
   along the opening axis, range 0..0.055 m, actuator ctrlrange matched.
2. **Finger collisions: mesh -> vertical boxes.** The printed finger meshes
   are disabled for collision (`contype=0 conaffinity=0`, visuals unchanged)
   and replaced by two box geoms (`finger_fixed`, `finger_moving`) with
   vertical inner faces, high friction (2.0) and stiff contact
   (`solref 0.004 1`). This gives a clean horizontal pinch instead of
   edge contacts on slanted mesh faces.
3. **wrist_flex range widened to +/-102 deg** (official +/-95). A straight-
   down gripper at table height needs ~97-98 deg; without this the grasp
   pose is unreachable and commands get clamped.

## In `examples/tennis_pick/simulation.py` (runtime, not in the XML)

4. **Servo damping / force limits.** The official position gains are
   underdamped (~30 Hz ring) which shakes a held ball out, and the light jaw
   slams shut in ~0.1 s regardless of any command ramp. We add velocity
   damping (kv=20) to all servos and a soft gain (kp=300) + damping to the
   gripper so closing is slow and force-limited. The gripper force limit is
   only 0.5 N: holding is done by the kinematic carry (item 5), so the low
   limit exists to make a *missed* grasp nudge the ball gently instead of
   launching it off the table. NOTE: a MuJoCo position actuator's force is
   `gainprm[0]*ctrl + biasprm[1]*qpos`, so both the gain and the bias must
   carry the new kp (and kv in gainprm[1]/biasprm[2]).
5. **Grasp = kinematic carry.** Bare jaw friction is unreliable for spheres,
   so `grasp()` additionally checks whether a ball centre lies inside the jaw
   capture region and, if so, carries that ball kinematically with the
   gripper (position/orientation written each step, ball collision disabled
   while carried). `release()` restores dynamics and inherits the carry
   velocity. This models a firm grip; the agent still must place the jaws
   around the ball using only cameras and encoders — no ball position is
   ever returned to the model.
6. **Arm collision friction raised to 1.2.**

## In `examples/tennis_pick/scene.xml`

7. **Tennis balls scaled to radius 0.02 m** (4 cm diameter) instead of the
   regulation 6.7 cm: the SO-101 finger span (~4.4 cm) cannot enclose a
   regulation ball at table level without the fingers interfering with the
   table. Mass kept at 0.057 kg. Stiff, high-friction contact
   (`solref 0.004 1`, friction 1.2).
8. **Collection box walls lowered to 0.045 m** and box placed at
   (0.13, 0.16) m so the vertical-gripper workspace can drop a ball over the
   wall (a straight-down gripper cannot reach higher than ~0.10 m).
9. **`<size memory="200M">`** added: the default contact/collision arena is
   too small for this model's mesh broadphase and aborts with
   `mj_stackAlloc: out of memory ... mj_collideTree` on some ball layouts.
10. **No perception aids**: the table is plain (an earlier 5 cm checker grid
    and an occlusion-avoiding retracted start pose were removed so the model
    solves the task purely from camera images).
11. **Agent tools are async** so pydantic-ai executes them on the event-loop
    (main) thread: MuJoCo EGL contexts are thread-bound and rendering from
    worker threads segfaults mid-episode.
## Mesh storage

The official STL meshes (~19 MB) are **not** committed to git. `fetch_assets.py`
downloads them from the exact commit pinned in `SOURCE.json` via git
sparse-checkout (raw.githubusercontent.com is blocked on some networks).
`engine.py` raises a clear error pointing at the fetch script if they are
missing.
