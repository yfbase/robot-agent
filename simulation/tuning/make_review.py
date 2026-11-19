"""Build a self-contained HTML review page comparing one successful and one
failed LLM episode side by side (filmstrips + tool transcript + model's own
final words + ground-truth verdict). Camera frames are embedded as base64 so
the page works offline; the episode MP4s are linked by relative path.

Run:
    uv run python examples/tennis_pick/tuning/make_review.py \
        --success llm_stable/episode_00 --failure llm_stable/episode_02
"""

import argparse
import base64
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXPERIMENTS = HERE / "experiments"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def episode_block(rel: str, label: str, verdict_class: str) -> str:
    ep = EXPERIMENTS / rel
    ev = json.loads((ep / "evaluation.json").read_text())
    frames = sorted((ep / "images").glob("*.jpg"))
    strip = "".join(
        f'<figure><img src="data:image/jpeg;base64,{b64(f)}" alt="{f.name}">'
        f"<figcaption>{html.escape(f.stem)}</figcaption></figure>"
        for f in frames
    )
    rows = []
    for line in (ep / "tool_calls.jsonl").read_text().splitlines():
        r = json.loads(line)
        args = ", ".join(f"{k}={v}" for k, v in r["args"].items())
        rows.append(
            f"<tr><td>{r['t']:.2f}</td><td><code>{html.escape(r['tool'])}</code></td>"
            f"<td><code>{html.escape(args)}</code></td></tr>"
        )
    transcript = "<table><tr><th>t(s)</th><th>tool</th><th>args</th></tr>" + "".join(rows) + "</table>"
    final = html.escape((ep / "agent_final.txt").read_text())
    balls = ev["final_state"]["balls"]
    return f"""
<section class="ep {verdict_class}">
  <h2>{label} — {html.escape(rel)}</h2>
  <p class="verdict">seed {ev['seed']} · placed {ev['placed']}/{ev['total']} ·
     <strong>{'SUCCESS' if ev['success'] else 'FAILURE'}</strong> ·
     tool calls {ev['tool_calls']} · error {html.escape(str(ev['error']))}</p>
  <p class="truth">ground-truth final ball position(s): {html.escape(json.dumps(balls))}</p>
  <h3>Camera filmstrip (every frame the agent saw, in order)</h3>
  <div class="strip">{strip}</div>
  <h3>Episode video (relative link; open page from disk to play)</h3>
  <p><a href="{rel}/top_view.mp4">top_view.mp4</a> · <a href="{rel}/side_view.mp4">side_view.mp4</a></p>
  <h3>Tool-call transcript</h3>
  {transcript}
  <h3>Agent's final reply (verbatim)</h3>
  <pre>{final}</pre>
</section>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--success", default="llm_stable/episode_00")
    ap.add_argument("--failure", default="llm_stable/episode_02")
    ap.add_argument("--out", default="review_success_vs_failure.html")
    args = ap.parse_args()
    body = episode_block(args.success, "SUCCESSFUL LLM episode", "ok")
    body += episode_block(args.failure, "FAILED LLM episode", "bad")
    page = f"""<!doctype html><meta charset="utf-8">
<title>LLM pick-and-place: success vs failure</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 h1{{font-size:22px}} h2{{font-size:18px;margin-top:0}} h3{{font-size:14px;color:#9cf}}
 .ep{{border:1px solid #444;border-radius:8px;padding:16px;margin:16px 0;background:#1a1a1a}}
 .ep.ok{{border-color:#2c6}} .ep.bad{{border-color:#c33}}
 .verdict{{font-size:14px}} .truth{{font-size:12px;color:#aaa}}
 .strip{{display:flex;flex-wrap:wrap;gap:8px}}
 figure{{margin:0}} figure img{{width:220px;border:1px solid #333;border-radius:4px}}
 figcaption{{font-size:10px;color:#888;text-align:center}}
 table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid #333;padding:3px 6px}}
 pre{{white-space:pre-wrap;font-size:12px;background:#000;padding:10px;border-radius:6px;max-height:420px;overflow:auto}}
 a{{color:#9cf}}
</style>
<h1>LLM-driven SO-101 tennis pick-and-place — one success vs one failure</h1>
<p>Same campaign (<code>llm_stable</code>), same prompt/config; different random ball layout.
Frames are exactly what the model received; transcripts and final replies are verbatim.</p>
{body}
"""
    out = EXPERIMENTS / args.out
    out.write_text(page)
    print("wrote", out)


if __name__ == "__main__":
    main()