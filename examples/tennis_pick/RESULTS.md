# Experiment results — LLM-driven SO-101 tennis-ball pick-and-place

All runs are archived under `experiments/<run_id>/` with full records
(config, scene, model source + modifications, per-episode images, tool-call
log, conversation, ground-truth evaluation, joint trajectory, dual-camera
video). Replay any episode with `simulation/replay.py`.

## Physics baseline (scripted reference policy, no LLM)

The reference policy (`simulation/tuning/tune_grasp_search.py`, also `--scripted`)
validates that the scene + grasp abstraction can solve the task. It succeeds
on every seed tested, including seeds that previously crashed the simulator:

| run | seeds | success |
| --- | ----- | ------- |
| tune_grasp_search sweep | 7,11,3,21,5,42 | 6/6 |
| baseline_scripted | 7,8 | 2/2 |
| baseline_check / check2 | 7 / 7,8 | 1/1, 2/2 |
| baseline_777 / 777b | 777 / 777,778 | 1/1, 2/2 |

## LLM campaigns (deepseek-v4-flash, vision)

Progression of prompt / infrastructure improvements. Early runs failed for
*infrastructure* reasons, not model capability:

| run | config | outcome | note |
| --- | ------ | ------- | ---- |
| llm_run1 | v1 prompt | 0/3 | bug: raw bytes in prompt (fixed) |
| llm_run2 | v1 prompt | 0/3 | bug: EGL context across threads (fixed) |
| llm_run3 | v1 free exploration | 0/1 | hit 40-request limit wandering in joint space |
| llm_run7_campaign | v2 lookup tables, 25 req | 1/3 | 2 episodes hit 25-req limit / arena crash |
| llm_run9b | v2.1 + grid + retracted ready, 40 req | 1/2 | ep1 arena crash (since fixed) |
| llm_debug1 | v2.1 + grid + retracted ready | 1/1 | clean first-try success, 18 tools |
| llm_final | + 200M arena + carried experience | 1/2 | ep2 mid-episode EGL segfault (since fixed by async tools) |
| llm_stable | + async tools (main-thread render) | 1/3 | no crashes; 2 episodes lost the ball after a missed grasp nudged it away |
| llm_gentle | + 0.5 N gripper limit + wider FOV | aborted | exploratory; wider FOV shrank the grid and hurt coordinate reading, so FOV reverted to 40 |
| llm_pure_vision | **perception aids removed** (plain table, neutral start pose) | 1/3 | pure-vision variant per user directive; no grid / no occlusion-avoiding pose |
| llm_pure_v2 | + A1 scale self-calibration, A2 occlusion-free parking, A3 miss re-localisation, A4 side-view straddle gate, B7 early-stop on empty grasp | 1/3 | ep0 improved 57-tool-fail -> 19-tool-success; ep1/ep2 still fail because the model's recovery/park arm sweeps bat the ball off the table (not grasp force). Next: lift-then-pan parking |
| llm_pure_v3_fast | + lift-then-pan (never sweep at ball height) + image-history pruning + live-video off + tighter settle | **3/3** | wall 263/110/210 s (avg ~194 s) vs v2 307/1649/1034 s; lift-then-pan removed the recovery-sweep ball-loss failure |

Aggregate over completed LLM episodes of the final configuration
(debug1 + run9b + llm_final + llm_stable): **4/8 = 50%**, with zero simulator
crashes once the async-tool and arena fixes landed. The scripted reference
policy remains 100% on all seeds tested.

## What made it work

1. **Grasp physics** (see `simulation/so101_model/MODIFICATIONS.md`): parallel-slide
   gripper + vertical box finger faces + compliant force-limited close +
   kinematic carry. Scripted baseline 6/6.
2. **Perception aids (later REMOVED)**: an earlier 5 cm table grid and a
   retracted start pose helped coordinate reading / occlusion. Per user
   directive these were removed so the model solves the task purely from
   camera images; see `llm_pure_vision` (1/3) vs aided runs. Scale must now be
   inferred from known object sizes (ball 0.02 m, box inner 0.12 m).
3. **Lookup-table prompt**: hover + grasp joint tables vs radius, plus an
   explicit per-ball recipe (release → hover → grasp → lift → box → release →
   verify). Removes joint-space trial-and-error.
4. **Experience loop**: after every episode (success *or* failure) the agent
   writes EXPERIENCE NOTES; the runner feeds them into the next episode and
   can carry them across runs (`--experience-from`). Notes visibly improve
   aiming (e.g. "interpolate calibration rows", "20-35 mm opening = secure
   hold", "re-image at hover before descending").
5. **Stability fixes**: thread-local MuJoCo renderers (pydantic-ai runs tools
   in worker threads); `<size memory="200M">` (default collision arena aborts
   with `mj_stackAlloc` overflow on some layouts); adequate request budget
   (40) so retries can complete.

## Per-episode wall time (llm_pure_v3_fast)

| episode | tools | sim s | wall s | LLM+overhead s |
| --- | --- | --- | --- | --- |
| ep0 | 31 | 11.9 | 262.9 | 251.0 |
| ep1 | 21 | 10.1 | 109.7 | 99.6 |
| ep2 | 28 | 10.3 | 210.1 | 199.8 |

Wall time is dominated by LLM requests (each re-sends the conversation); MuJoCo
sim seconds are a small fraction. Latency optimizations: image-history pruning
(keep 2 newest frames), live video off by default (replay on demand), tighter
servo settle. `evaluation.json:wall_seconds` and `tool_calls.jsonl:wall` enable
attribution. Earlier aided/pure runs: success ~310-320 s, failures 640-1650 s.

## Honest limitations

* Success is not 100%: when the model mis-estimates a ball's radius/pan the
  jaws miss and can knock the ball to a new (sometimes arm-occluded) position,
  costing retries. With a 40-request budget most episodes recover; a few
  exhaust the budget.
* The grasp uses a documented kinematic-carry abstraction rather than pure
  friction; ball size is scaled to 4 cm and box walls lowered to 4.5 cm so the
  task is physically reachable for this gripper. These are fidelity trade-offs
  listed in MODIFICATIONS.md, not hidden cheats: the agent still receives only
  camera images and encoder angles.