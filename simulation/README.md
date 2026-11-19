# `simulation/` — MuJoCo SO-101 tennis-ball pick-and-place simulation

The simulation half of the task. The agent half (tools / prompt / runner)
lives in `examples/tennis_pick/` and imports this package as `simulation`.

## Contents

| path | purpose |
| ---- | ------- |
| `engine.py` | `TennisPickSim`: scene load, reset/random balls, servo read/write, grasp/release, cameras, scoring, video + trajectory recording |
| `__init__.py` | re-exports the public API (`TennisPickSim`, servo names, scene constants) |
| `scene.xml` | table + fixed-base arm + box + balls + top/side cameras + 5 cm grid |
| `so101_model/` | official TheRobotStudio SO-101 MJCF (`SOURCE.json` pins the commit) + `MODIFICATIONS.md`; the ~19 MB STL meshes are **not** in git — fetch with `fetch_assets.py` |
| `calibration.json`, `calibration_hover.json` | radius → joint-angle lookup tables (grasp / hover height) embedded in the agent prompt |
| `tuning/` | offline probes & sweeps that derived the grasp placement and servo settings; `tune_grasp_search.py` = reference policy; `make_review.py` = success/failure review page |
| `experiments/` | generated run records (git-ignored; on disk for review). Includes `deliver_videos/` and `review_success_vs_failure.html` |
| `replay.py` | deterministic frame-by-frame replay of a recorded `trajectory.npz` |

## Quick use

```bash
# one-time: fetch the pinned official STL meshes (not stored in git)
uv run python simulation/so101_model/fetch_assets.py

MUJOCO_GL=egl uv run python simulation/replay.py simulation/experiments/<run>/episode_00
MUJOCO_GL=egl uv run python simulation/tuning/render_scene.py      # layout check
MUJOCO_GL=egl uv run python simulation/tuning/make_review.py       # review page
```

The agent campaign is launched from `examples/tennis_pick/run_experiment.py`;
it writes records into `simulation/experiments/`.

## Fidelity

Read `so101_model/MODIFICATIONS.md` before judging physical realism: the
gripper is modeled as a parallel slide jaw with box finger faces and a
kinematic grasp carry; balls are scaled to 4 cm; box walls lowered to 4.5 cm;
servo damping / force limits and a 200 M collision arena are set. The agent
never receives ground-truth state — only camera JPEGs and encoder angles.