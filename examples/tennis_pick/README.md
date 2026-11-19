# Tennis-ball pick-and-place with an LLM-driven SO-101 arm (MuJoCo)

An application built on the `robot_agent` framework (pydantic-ai). A vision
LLM controls a fixed-base **SO-101 (SO-ARM101)** arm in MuJoCo to pick tennis
balls off a table and drop them into a blue box. The framework in
`robot_agent/` is untouched.

## Layout (split between simulation and agent)

* `simulation/` (repo root) — the MuJoCo simulation itself: `engine.py`
  (`TennisPickSim`), `scene.xml`, `so101_model/` (official model + local
  modifications), `calibration*.json`, `tuning/` probes, `experiments/`
  records, and `replay.py`. Importable as the `simulation` package.
* `examples/tennis_pick/` (here) — the agent-facing application: the six
  whitelisted tools (`agent_tools.py`), the prompt (`prompts.py`), and the
  campaign runner (`run_experiment.py`). It imports `simulation`.

## Scene

* Table top at world height 0; the arm base is **bolted** at the origin (the
  official SO-101 base body has no joint, so it cannot move).
* Open-top blue box at (0.13, 0.16) m, inner 0.12 x 0.12 m, wall 0.045 m.
* Up to 3 tennis balls (radius 0.02 m, see fidelity notes) spawned at random
  positions in front of the base each episode (`--seed` controls the layout).
* Two fixed cameras: `top_view` (straight down) and `side_view` (horizontal).
* Perception aids: the table top carries a 5 cm checker grid so the model can
  read world coordinates by counting squares, and the start pose retracts the
  arm to the side (shoulder_pan=90) so it does not occlude the balls in the
  top view.

## Agent capabilities (the ONLY tools)

| tool | returns |
| ---- | ------- |
| `get_top_view` | overhead JPEG image |
| `get_side_view` | side JPEG image |
| `get_servo_angles` | 5 arm angles (deg) + gripper opening (mm) |
| `set_servo_angles` | command arm angles (deg), wait for settle, return achieved |
| `grasp` | close gripper (pre-tuned), return achieved angles |
| `release` | open gripper (pre-tuned), return achieved angles |

No object positions, forces or simulator state are ever exposed to the model.
Grasp success is decided inside the simulation (a gripper "knows" it holds
something); the model only sees cameras and encoders.

Between episodes the agent writes **EXPERIENCE NOTES** (its own reflection on
what worked / failed); the runner saves them to `experience.md` and feeds them
into the next episode's prompt.

## Run

```bash
# one-time prerequisite: fetch the pinned official SO-101 STL meshes
uv run python simulation/so101_model/fetch_assets.py

# real LLM (needs API_KEY / BASE_URL / MODEL in .env)
MUJOCO_GL=egl uv run python examples/tennis_pick/run_experiment.py \
    --episodes 3 --seed 20261119 --balls 1

# carry experience notes from a previous run into a new campaign
MUJOCO_GL=egl uv run python examples/tennis_pick/run_experiment.py \
    --episodes 4 --seed 777 --max-requests 40 \
    --experience-from simulation/experiments/<prev>/experience.md

# physics baseline / smoke test without an API key (scripted reference policy)
MUJOCO_GL=egl uv run python examples/tennis_pick/run_experiment.py --scripted

# replay a recorded episode exactly (no LLM, no re-simulation)
MUJOCO_GL=egl uv run python simulation/replay.py \
    simulation/experiments/<run>/episode_00
```

`MUJOCO_GL=egl` selects offscreen rendering (works headless via Mesa).

## Experiment records (for review / reproduction)

Each run creates `simulation/experiments/<run_id>/`:

* `run_config.json` — seed, episodes, balls, model, base URL, limits, MuJoCo
  version, git commit, timestamp.
* `scene.xml`, `SOURCE.json`, `MODIFICATIONS.md` — exact scene + model source
  and every local fidelity modification.
* `experience.md` — the agent's accumulated notes (final state).
* `summary.json` — per-episode success / placed counts.
* `episode_XX/`
  * `images/` — every camera frame the agent saw (numbered), plus initial views.
  * `tool_calls.jsonl` — every tool call with args, result summary, sim time.
  * `messages.json` — full pydantic-ai conversation (ModelMessagesTypeAdapter).
  * `evaluation.json` — harness-side ground-truth scoring + final state.
  * `trajectory.npz` — qpos/ctrl per physics step + initial state + seed.
  * `top_view.mp4`, `side_view.mp4` — rendered episode video.
  * `agent_final.txt`, `experience_before.md`.

## Fidelity notes

The official SO-101 MJCF is used for link kinematics, masses and visuals.
Several simulation-level abstractions were required to make sphere grasping
stable (parallel-slide gripper, box finger collisions, kinematic grasp carry,
servo damping, scaled ball, lowered box wall, widened wrist_flex). Every one
is documented in `simulation/so101_model/MODIFICATIONS.md`. Read it before judging
physical realism.

## Offline tuning scripts

`simulation/tuning/` holds the probes and sweeps used to derive the grasp placement and
servo settings (kinematics probe, jaw-face measurement, contact-normal debug,
joint-space pose search, scripted grasp sweeps). `tune_grasp_search.py`
contains the reference policy and the pose search reused by `--scripted`.
`simulation/calibration.json` is the vertical-gripper joint table embedded in the prompt.