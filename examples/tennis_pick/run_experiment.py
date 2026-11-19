"""Run the LLM agent on the MuJoCo tennis-ball pick-and-place task.

Everything the agent can do is whitelisted in agent_tools.py (two cameras,
servo read/write, grasp, release). The runner scores episodes from simulator
state (harness-side only), archives full experiment records for replay, and
lets the agent write EXPERIENCE NOTES that are fed back into later episodes.

Run (needs API_KEY in .env):
    MUJOCO_GL=egl uv run python examples/tennis_pick/run_experiment.py \
        --episodes 3 --seed 20261119 --balls 1

Offline smoke test (no API key, uses a scripted stand-in, not the LLM):
    MUJOCO_GL=egl uv run python examples/tennis_pick/run_experiment.py --scripted
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import UsageLimits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / "simulation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM / "tuning"))
sys.path.insert(0, str(HERE))  # agent_tools / prompts live beside this script

from agent_tools import make_tools
from prompts import reflection_message, system_prompt, task_message
from simulation import TennisPickSim


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def scripted_episode(sim: TennisPickSim) -> str:
    """Reference policy used by --scripted and as a physics baseline."""
    import numpy as np
    from simulation import BOX_CENTER_XY
    from tune_grasp_search import HEIGHT, LATERAL, search_pose

    for i in range(sim.num_balls):
        ball = sim.ball_position(i)
        pan = np.arctan2(-ball[1], ball[0])
        radial = np.array([np.cos(pan), -np.sin(pan), 0.0])
        gs = ball - radial * LATERAL + np.array([0.0, 0.0, HEIGHT])
        sim.grasp()
        p, _ = search_pose(sim, np.array([0.144, 0.0, 0.095]))
        sim.set_servo_angles(p)
        p, _ = search_pose(sim, gs + np.array([0.0, 0.0, 0.06]))
        sim.set_servo_angles(p)
        sim.release()
        for dz in (0.03, 0.015, 0.007, 0.0):
            p, _ = search_pose(sim, gs + np.array([0.0, 0.0, dz]))
            sim.set_servo_angles(p)
        sim.grasp()
        p, _ = search_pose(sim, gs + np.array([0.0, 0.0, 0.06]))
        sim.set_servo_angles(p)
        bp = np.arctan2(-BOX_CENTER_XY[1], BOX_CENTER_XY[0])
        br = np.array([np.cos(bp), -np.sin(bp), 0.0])
        bs = np.array([*BOX_CENTER_XY, 0.09]) - br * LATERAL
        p, _ = search_pose(sim, bs)
        sim.set_servo_angles(p)
        sim.release()
        p, _ = search_pose(sim, bs + np.array([0.0, 0.0, 0.01]))
        sim.set_servo_angles(p)
    return "DONE (scripted reference policy)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--balls", type=int, default=1)
    ap.add_argument("--max-requests", type=int, default=40)
    ap.add_argument("--max-tool-calls", type=int, default=120)
    ap.add_argument("--scripted", action="store_true", help="no LLM; reference policy")
    ap.add_argument(
        "--video",
        action="store_true",
        help="record live dual-camera MP4 during the run (slower); otherwise "
        "generate videos on demand from trajectory.npz via simulation/replay.py",
    )
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument(
        "--experience-from",
        type=Path,
        default=None,
        help="path to a previous run's experience.md to seed this run's notes",
    )
    args = ap.parse_args()

    run_id = args.run_dir or time.strftime("%Y%m%d_%H%M%S")
    run_dir = SIM / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scene.xml").write_text((SIM / "scene.xml").read_text())
    for name in ("SOURCE.json", "MODIFICATIONS.md"):
        shutil.copy(SIM / "so101_model" / name, run_dir / name)
    config = {
        "run_id": run_id,
        "seed": args.seed,
        "episodes": args.episodes,
        "balls_per_episode": args.balls,
        "scripted": args.scripted,
        "max_requests": args.max_requests,
        "max_tool_calls": args.max_tool_calls,
        "mujoco_version": mujoco.__version__,
        "git_commit": git_commit(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not args.scripted:
        from robot_agent.config import Settings

        settings = Settings()
        config["model"] = settings.model
        config["base_url"] = settings.base_url
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    experience_path = run_dir / "experience.md"
    if args.experience_from is not None and args.experience_from.is_file():
        experience_path.write_text(args.experience_from.read_text())
    else:
        experience_path.write_text("")
    summary = []

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        ep_dir = run_dir / f"episode_{ep:02d}"
        ep_dir.mkdir(exist_ok=True)
        sim = TennisPickSim(num_balls=args.balls, seed=ep_seed, record_video=args.video)
        if args.video:
            sim.start_video(ep_dir)
        ep_wall_start = time.time()
        tool_log = []

        def log(name, a, summary_str, _sim=sim, _log=tool_log):
            _log.append(
                {
                    "t": round(_sim.data.time, 3),
                    "wall": round(time.time(), 2),
                    "tool": name,
                    "args": a,
                    "result": summary_str,
                }
            )

        tools = make_tools(sim, ep_dir / "images", log)
        initial = [
            BinaryContent(data=sim.capture_jpeg("top_view"), media_type="image/jpeg"),
            BinaryContent(data=sim.capture_jpeg("side_view"), media_type="image/jpeg"),
        ]
        (ep_dir / "images").mkdir(exist_ok=True)
        (ep_dir / "images" / "0000_top_view.jpg").write_bytes(initial[0].data)
        (ep_dir / "images" / "0000_side_view.jpg").write_bytes(initial[1].data)
        (ep_dir / "experience_before.md").write_text(experience_path.read_text())

        print(f"\n=== episode {ep} (seed {ep_seed}) ===", flush=True)
        final_text = ""
        error = None
        if args.scripted:
            final_text = scripted_episode(sim)
        else:
            from pydantic_ai.models.openai import OpenAIResponsesModel
            from pydantic_ai.providers.openai import OpenAIProvider

            from robot_agent import RobotAgent
            from robot_agent.config import Settings

            from model_pruning import ImagePruningModel

            settings = Settings()
            base_model = OpenAIResponsesModel(
                settings.model,
                provider=OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key),
            )
            agent = RobotAgent(
                model=ImagePruningModel(base_model, keep_images=2),
                instructions=system_prompt(),
                tools=tools,
            )
            limits = UsageLimits(
                request_limit=args.max_requests, tool_calls_limit=args.max_tool_calls
            )
            result_messages = None
            try:
                result = agent.agent.run_sync(
                    [task_message(experience_path.read_text(), args.balls), *initial],
                    usage_limits=limits,
                )
                final_text = result.output
                result_messages = result.all_messages()
                (ep_dir / "messages.json").write_bytes(
                    ModelMessagesTypeAdapter.dump_json(result_messages)
                )
            except Exception as exc:  # keep the campaign alive on API errors
                error = f"{type(exc).__name__}: {exc}"
                print("episode error:", error, flush=True)

        if args.video:
            sim.finish_video()
        sim.save_trajectory(ep_dir / "trajectory.npz")
        placed = int(sum(sim.ball_in_box(i) for i in range(args.balls)))
        evaluation = {
            "episode": ep,
            "seed": ep_seed,
            "placed": placed,
            "total": args.balls,
            "success": placed == args.balls,
            "wall_seconds": round(time.time() - ep_wall_start, 1),
            "final_state": sim.state_snapshot(),
            "agent_final": final_text,
            "error": error,
            "tool_calls": len(tool_log),
        }
        (ep_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        # Reflection runs even on failure / usage-limit so experience accrues.
        if not args.scripted:
            from robot_agent import RobotAgent

            transcript = "\n".join(
                f"{row['tool']}({row['args']}) -> {row['result']}" for row in tool_log[-40:]
            )
            refl_prompt = (
                reflection_message(evaluation["success"], placed, args.balls)
                + "\n\nYour tool-call transcript this episode:\n"
                + transcript
            )
            refl_prompt += (
                "\n\nYour previous EXPERIENCE NOTES (update them, keep useful parts):\n"
                + (ep_dir / "experience_before.md").read_text()
            )
            try:
                refl_agent = RobotAgent(instructions=system_prompt())
                refl = refl_agent.chat_sync(
                    refl_prompt,
                )
                experience_path.write_text(refl.output)
                (ep_dir / "experience_after.md").write_text(refl.output)
            except Exception as exc:
                print("reflection error:", exc, flush=True)
        (ep_dir / "tool_calls.jsonl").write_text("\n".join(json.dumps(row) for row in tool_log))
        (ep_dir / "agent_final.txt").write_text(final_text)
        summary.append(evaluation)
        print(
            f"episode {ep}: placed {placed}/{args.balls} "
            f"success={evaluation['success']} tools={len(tool_log)}",
            flush=True,
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ok = sum(s["success"] for s in summary)
    print(f"\nRUN {run_id}: {ok}/{len(summary)} episodes successful")
    print("records:", run_dir)


if __name__ == "__main__":
    main()
