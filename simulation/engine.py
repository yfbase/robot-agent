"""MuJoCo simulation wrapper for the tennis-ball pick-and-place task.

Scene: an SO-101 arm (official TheRobotStudio model) bolted to a table at the
origin, an open-top collection box, up to 3 tennis balls, and two fixed
cameras (overhead ``top_view`` and lateral ``side_view``).

This module is shared by the offline tuning scripts and the agent tools.
It exposes two kinds of methods:

* Agent-facing: servo angles (degrees), camera frames, grasp/release.
  These return only what a real robot's encoders and cameras would return.
* Harness-facing: ``reset``, ``ball_in_box``, ``state_snapshot``, video and
  trajectory recording. Used by the runner for scoring and experiment logs;
  never exposed to the model.
"""

import threading
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE_PATH = HERE / "scene.xml"

ARM_SERVOS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_SERVOS = ARM_SERVOS + ["gripper"]

# Gripper setpoints in meters of jaw opening (the gripper is modeled as a
# parallel slide jaw, see so101_model/MODIFICATIONS.md). The agent never
# chooses a gripper setpoint; grasp()/release() use these fixed values.
GRIPPER_OPEN_M = 0.05
GRIPPER_CLOSE_M = 0.0
# Soft gripper servo stiffness: force = kp * position error, so the close is
# force-limited and compliant; the jaw stalls on the ball with a gentle
# squeeze instead of crushing it.
GRIPPER_KP = 300.0

# Comfortable start pose: gripper pointing straight down above the table.
# Note: shoulder_lift + elbow_flex + wrist_flex = +90 deg makes the gripper
# point vertically down (all three positive angles pitch the tip downward).
READY_POSE_DEG = {
    "shoulder_pan": 90.0,
    "shoulder_lift": -40.0,
    "elbow_flex": 35.0,
    "wrist_flex": 95.0,
    "wrist_roll": 0.0,
}

# Ball spawn region on the table (meters, world frame; table top is z = 0).
# Kept inside the vertical-gripper reachable envelope (radius <= 0.26 m).
SPAWN_X = (0.13, 0.26)
SPAWN_Y = (-0.15, 0.15)
SPAWN_RADIUS = (0.12, 0.26)
BALL_RADIUS = 0.02
BOX_CENTER_XY = (0.13, 0.16)  # keep in sync with scene.xml
BOX_INNER_HALF = 0.06
BOX_WALL_TOP = 0.045

CAMERAS = ("top_view", "side_view")
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
VIDEO_FPS = 25  # playback speed; frames sampled every 1/VIDEO_FPS sim-seconds


class TennisPickSim:
    """One MuJoCo instance of the task, with recording built in."""

    def __init__(self, num_balls: int = 1, seed: int = 0, record_video: bool = True):
        if not 1 <= num_balls <= 3:
            raise ValueError("num_balls must be 1..3")
        self.num_balls = num_balls
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if not any((HERE / "so101_model" / "assets").glob("*.stl")):
            raise RuntimeError(
                "SO-101 STL meshes are missing (they are not stored in git). "
                "Fetch them with: uv run python simulation/so101_model/fetch_assets.py"
            )
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.record_video = record_video

        self._qposadr = {j: self.model.joint(j).qposadr[0] for j in ALL_SERVOS}
        self._ctrladr = {j: self.model.actuator(j).id for j in ALL_SERVOS}
        self._limits_deg = {j: np.rad2deg(self.model.joint(j).range).copy() for j in ARM_SERVOS}
        # Gripper limits are reported in millimeters of opening.
        self._limits_deg["gripper"] = self.model.joint("gripper").range.copy() * 1000.0
        self._ball_qposadr = [
            self.model.joint(f"tennis_ball_{i}_free").qposadr[0] for i in range(3)
        ]
        # Grippy fingers: raise sliding friction on the arm's collision geoms
        # (the official model leaves them at the MuJoCo default of 1.0).
        arm_bodies = {
            self.model.body(n).id
            for n in (
                "base",
                "shoulder",
                "upper_arm",
                "lower_arm",
                "wrist",
                "gripper",
                "moving_jaw_so101_v1",
            )
        }
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] in arm_bodies and self.model.geom_contype[g] > 0:
                self.model.geom_friction[g, 0] = 1.2
        # Slow, well-damped arm servos: the official gains are underdamped
        # (~30 Hz ring) which shakes a held ball out of the jaws.
        for j in ARM_SERVOS:
            self.model.actuator_gainprm[self._ctrladr[j], 1] = 20.0
            self.model.actuator_biasprm[self._ctrladr[j], 2] = -20.0
        # Compliant gripper servo (see GRIPPER_KP). A position actuator's
        # force is gainprm[0]*ctrl + biasprm[1]*qpos, so BOTH must carry kp.
        self.model.actuator_gainprm[self._ctrladr["gripper"], 0] = GRIPPER_KP
        self.model.actuator_biasprm[self._ctrladr["gripper"], 1] = -GRIPPER_KP
        # Velocity damping + small force limit: without these the light jaw
        # slams shut in ~0.1 s regardless of the command ramp, impacting the
        # ball out of the grip.
        self.model.actuator_gainprm[self._ctrladr["gripper"], 1] = 20.0
        self.model.actuator_biasprm[self._ctrladr["gripper"], 2] = -20.0
        self.model.actuator_forcerange[self._ctrladr["gripper"]] = [-0.5, 0.5]
        # EGL/OSMesa contexts are thread-bound and pydantic-ai runs tools in
        # worker threads, so each thread gets its own renderer lazily.
        self._renderers_local = threading.local()
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_paths: dict[str, Path] = {}
        self._next_frame_time = 0.0
        self._carried: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._carry_vel: dict[int, np.ndarray] = {}
        self._trajectory: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._initial_state: dict | None = None
        self.reset()

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        """Start a fresh task: ready pose, open gripper, new random ball layout."""
        mujoco.mj_resetData(self.model, self.data)
        self._carried = {}
        self._carry_vel = {}
        for i in range(3):
            self.model.geom_contype[self.model.geom(f"tennis_ball_{i}").id] = 1
            self.model.geom_conaffinity[self.model.geom(f"tennis_ball_{i}").id] = 1
        for name, deg in READY_POSE_DEG.items():
            self.data.qpos[self._qposadr[name]] = np.deg2rad(deg)
            self.data.ctrl[self._ctrladr[name]] = np.deg2rad(deg)
        self.data.ctrl[self._ctrladr["gripper"]] = GRIPPER_OPEN_M
        self.data.qpos[self._qposadr["gripper"]] = GRIPPER_OPEN_M

        positions = self._sample_ball_positions(self.num_balls)
        for i in range(3):
            adr = self._ball_qposadr[i]
            weld_id = self.model.equality(f"park_ball_{i}").id
            if i < self.num_balls:
                x, y = positions[i]
                self.data.qpos[adr : adr + 3] = [x, y, BALL_RADIUS + 0.02]
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
                self.data.eq_active[weld_id] = 0
            else:
                # Park unused balls out of camera sight, welded to the world.
                self.data.qpos[adr : adr + 3] = [-1.0, 0.2 * i, -0.35]
                self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
                self.data.eq_active[weld_id] = 1
        mujoco.mj_forward(self.model, self.data)

        self._trajectory = []
        self._initial_state = {
            "qpos": self.data.qpos.copy(),
            "ctrl": self.data.ctrl.copy(),
        }
        self._next_frame_time = 0.0
        self._settle(max_time=0.8, min_time=0.4)  # let balls drop onto the table

    def _sample_ball_positions(self, n: int) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        while len(points) < n:
            x = self.rng.uniform(*SPAWN_X)
            y = self.rng.uniform(*SPAWN_Y)
            radius = np.hypot(x, y)
            if not SPAWN_RADIUS[0] <= radius <= SPAWN_RADIUS[1]:
                continue
            if (
                abs(x - BOX_CENTER_XY[0]) < BOX_INNER_HALF + 0.05
                and abs(y - BOX_CENTER_XY[1]) < BOX_INNER_HALF + 0.05
            ):
                continue
            if any(np.hypot(x - px, y - py) < 2.5 * BALL_RADIUS for px, py in points):
                continue
            points.append((x, y))
        return points

    # --------------------------------------------------------- agent-facing

    def servo_angles_deg(self) -> dict[str, float]:
        """Current servo angles in degrees, as read from joint encoders."""
        angles = {
            j: round(float(np.rad2deg(self.data.qpos[self._qposadr[j]])), 2) for j in ARM_SERVOS
        }
        angles["gripper"] = round(float(self.data.qpos[self._qposadr["gripper"]] * 1000.0), 2)
        return angles

    def servo_limits_deg(self) -> dict[str, tuple[float, float]]:
        return {j: (float(lo), float(hi)) for j, (lo, hi) in self._limits_deg.items()}

    def set_servo_angles(self, angles_deg: dict[str, float]) -> dict[str, float]:
        """Command the five arm servos (degrees) and wait until motion settles.

        Unknown names raise; out-of-range values are clamped to joint limits.
        The gripper servo is not touched here. Returns the achieved angles.
        """
        for name in angles_deg:
            if name not in ARM_SERVOS:
                raise ValueError(f"unknown or non-arm servo: {name}")
        for name, deg in angles_deg.items():
            lo, hi = self._limits_deg[name]
            clipped = float(np.clip(deg, lo, hi))
            self.data.ctrl[self._ctrladr[name]] = np.deg2rad(clipped)
        self._settle()
        return self.servo_angles_deg()

    def grasp(self) -> dict[str, float]:
        """Close the parallel jaws onto whatever is between them, then hold.

        Compliant close: the command ramps toward fully closed at a limited
        speed with a soft servo gain; once the jaw stalls on an object we
        stop ramping and hold a small squeeze (2 mm) past the stall point.
        Driving all the way to fully closed would crush the ball out.
        """
        gid = self._ctrladr["gripper"]
        qadr = self._qposadr["gripper"]
        start_m = float(self.data.ctrl[gid])
        t0 = self.data.time
        stalled_since: float | None = None
        while self.data.time - t0 < 2.5:
            opening = max(GRIPPER_CLOSE_M, start_m - 0.03 * (self.data.time - t0))
            self.data.ctrl[gid] = opening
            self._record_step()
            self._step()
            q = float(self.data.qpos[qadr])
            v = float(abs(self.data.qvel[self.model.joint("gripper").dofadr[0]]))
            if q < 0.045 and v < 0.002:
                stalled_since = self.data.time if stalled_since is None else stalled_since
                if self.data.time - stalled_since >= 0.08:
                    break
            else:
                stalled_since = None
        self.data.ctrl[gid] = max(GRIPPER_CLOSE_M, float(self.data.qpos[qadr]) - 0.008)
        self._settle()
        self._attach_balls_in_jaws()
        return self.servo_angles_deg()

    def _attach_balls_in_jaws(self) -> None:
        """Grasp abstraction: any ball whose center lies inside the jaw
        capture region is carried kinematically with the gripper (models a
        firm grip; bare jaw friction is unreliable for spheres). The agent
        still has to place the jaws around the ball using only cameras and
        encoders. See MODIFICATIONS.md."""
        opening = float(self.data.qpos[self._qposadr["gripper"]])
        gx = self.data.body("gripper").xpos
        gm = self.data.body("gripper").xmat.reshape(3, 3)
        for i in range(self.num_balls):
            p = gm.T @ (self.ball_position(i) - gx)
            inside = (
                -0.013 < p[0] < -0.006 + opening + 0.006
                and abs(p[1]) < 0.022
                and -0.125 < p[2] < -0.050
            )
            if inside and i not in self._carried:
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(
                    quat,
                    np.ascontiguousarray(
                        gm.T @ self.data.body(f"tennis_ball_{i}").xmat.reshape(3, 3)
                    ).flatten(),
                )
                self._carried[i] = (p.copy(), quat)
                self.model.geom_contype[self.model.geom(f"tennis_ball_{i}").id] = 0
                self.model.geom_conaffinity[self.model.geom(f"tennis_ball_{i}").id] = 0

    def _detach_balls(self) -> None:
        for i in list(self._carried):
            self.model.geom_contype[self.model.geom(f"tennis_ball_{i}").id] = 1
            self.model.geom_conaffinity[self.model.geom(f"tennis_ball_{i}").id] = 1
            # Inherit the carry velocity so the drop looks natural.
            adr = self._ball_qposadr[i]
            self.data.qvel[adr : adr + 3] = self._carry_vel.get(i, np.zeros(3))
        self._carried = {}
        self._carry_vel = {}

    def _step(self) -> None:
        """One physics step, then carry any grasped ball with the gripper."""
        mujoco.mj_step(self.model, self.data)
        if not self._carried:
            return
        gx = self.data.body("gripper").xpos
        gm = self.data.body("gripper").xmat.reshape(3, 3)
        gq = np.zeros(4)
        mujoco.mju_mat2Quat(gq, np.ascontiguousarray(gm).flatten())
        for i, (off, offq) in self._carried.items():
            adr = self._ball_qposadr[i]
            new_pos = gx + gm @ off
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, gq, offq)
            vel = (new_pos - self.data.qpos[adr : adr + 3]) / self.model.opt.timestep
            self._carry_vel[i] = vel
            self.data.qpos[adr : adr + 3] = new_pos
            self.data.qpos[adr + 3 : adr + 7] = new_quat
            self.data.qvel[adr : adr + 3] = vel
            self.data.qvel[adr + 3 : adr + 6] = 0.0

    def release(self) -> dict[str, float]:
        """Open the jaws to the pre-tuned release opening and wait."""
        gid = self._ctrladr["gripper"]
        start_m = float(self.data.ctrl[gid])
        t0 = self.data.time
        while self.data.time - t0 < 0.4:
            frac = (self.data.time - t0) / 0.4
            self.data.ctrl[gid] = start_m + (GRIPPER_OPEN_M - start_m) * frac
            self._record_step()
            self._step()
        self.data.ctrl[gid] = GRIPPER_OPEN_M
        self._detach_balls()
        self._settle()
        return self.servo_angles_deg()

    def capture(self, camera: str) -> np.ndarray:
        """Render one of the fixed cameras; returns an RGB uint8 array."""
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera: {camera}")
        renderers = getattr(self._renderers_local, "renderers", None)
        if renderers is None:
            renderers = {}
            self._renderers_local.renderers = renderers
        if camera not in renderers:
            renderers[camera] = mujoco.Renderer(self.model, FRAME_HEIGHT, FRAME_WIDTH)
        renderer = renderers[camera]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def capture_jpeg(self, camera: str) -> bytes:
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(self.capture(camera), cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    # ------------------------------------------------------- harness-facing

    def ball_position(self, index: int) -> np.ndarray:
        adr = self._ball_qposadr[index]
        return self.data.qpos[adr : adr + 3].copy()

    def ball_in_box(self, index: int) -> bool:
        """Scoring predicate: ball resting inside the box cavity."""
        x, y, z = self.ball_position(index)
        return (
            abs(x - BOX_CENTER_XY[0]) < BOX_INNER_HALF
            and abs(y - BOX_CENTER_XY[1]) < BOX_INNER_HALF
            and z < BOX_WALL_TOP
        )

    def state_snapshot(self) -> dict:
        return {
            "servo_angles_deg": self.servo_angles_deg(),
            "balls": [
                {
                    "position": self.ball_position(i).round(4).tolist(),
                    "in_box": bool(self.ball_in_box(i)),
                }
                for i in range(self.num_balls)
            ],
            "sim_time": float(self.data.time),
        }

    # ------------------------------------------------------------- stepping

    def _settle(self, max_time: float = 3.0, min_time: float = 0.3) -> None:
        """Step physics until the arm and balls are calm (or max_time hits)."""
        start = self.data.time
        calm_since: float | None = None
        arm_dofadr = [self.model.joint(j).dofadr[0] for j in ARM_SERVOS]
        while True:
            self._record_step()
            self._step()
            elapsed = self.data.time - start
            arm_speed = float(np.max(np.abs(self.data.qvel[arm_dofadr])))
            grip_speed = float(abs(self.data.qvel[self.model.joint("gripper").dofadr[0]]))
            ball_speed = max(
                float(
                    np.linalg.norm(
                        self.data.qvel[self._ball_qposadr[i] : self._ball_qposadr[i] + 3]
                    )
                )
                for i in range(self.num_balls)
            )
            calm = arm_speed < 0.03 and grip_speed < 0.005 and ball_speed < 0.03
            if calm and elapsed >= min_time:
                calm_since = self.data.time if calm_since is None else calm_since
                if self.data.time - calm_since >= 0.1:
                    break
            else:
                calm_since = None
            if elapsed >= max_time:
                break
        self._record_step()

    def _record_step(self) -> None:
        self._trajectory.append(
            (float(self.data.time), self.data.qpos.copy(), self.data.ctrl.copy())
        )
        if self.record_video and self.data.time >= self._next_frame_time:
            self._next_frame_time = self.data.time + 1.0 / VIDEO_FPS
            for cam in CAMERAS:
                writer = self._video_writers.get(cam)
                if writer is not None:
                    frame = cv2.cvtColor(self.capture(cam), cv2.COLOR_RGB2BGR)
                    writer.write(frame)

    # -------------------------------------------------------------- outputs

    def start_video(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            path = directory / f"{cam}.mp4"
            self._video_writers[cam] = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            self._video_paths[cam] = path

    def finish_video(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers = {}

    def save_trajectory(self, path: Path) -> None:
        """Save everything needed for deterministic offline replay."""
        times = np.array([row[0] for row in self._trajectory])
        qpos = np.array([row[1] for row in self._trajectory])
        ctrl = np.array([row[2] for row in self._trajectory])
        np.savez_compressed(
            path,
            time=times,
            qpos=qpos,
            ctrl=ctrl,
            initial_qpos=self._initial_state["qpos"],
            initial_ctrl=self._initial_state["ctrl"],
            timestep=self.model.opt.timestep,
            seed=self.seed,
            num_balls=self.num_balls,
        )
