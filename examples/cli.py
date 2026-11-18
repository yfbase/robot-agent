"""Example application: command-line chat with text and image attachments.

Usage:
    uv run python examples/cli.py                     # interactive chat
    uv run python examples/cli.py "hello"             # one-shot text message
    uv run python examples/cli.py --image cat.png "what is this?"

Interactive commands:
    /image <path>   attach an image to the next message
    /clear          forget the conversation history
    /exit           quit
"""

import argparse
import sys

from robot_agent import RobotAgent, Settings, load_image
from robot_agent.tools import capture_photo

HELP_TEXT = """\
Commands:
  /image <path>   attach an image to the next message
  /clear          forget the conversation history
  /exit           quit
Anything else is sent to the agent as a message.\
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="robot-agent",
        description="Chat with an OpenAI-compatible model (text + images).",
    )
    parser.add_argument("prompt", nargs="*", help="one-shot message; omit for interactive chat")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        metavar="PATH",
        help="attach an image to the first message (repeatable)",
    )
    parser.add_argument("--model", help="override the model name from settings")
    return parser.parse_args()


def _run_repl(agent: RobotAgent, pending_images: list[str]) -> None:
    print(HELP_TEXT)
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text in ("/exit", "/quit"):
            break
        if text == "/clear":
            agent.clear_history()
            pending_images.clear()
            print("History cleared.")
            continue
        if text.startswith("/image"):
            path = text[len("/image") :].strip().strip('"')
            if not path:
                print("Usage: /image <path>")
                continue
            try:
                load_image(path)
            except (FileNotFoundError, ValueError) as exc:
                print(exc)
                continue
            pending_images.append(path)
            print(f"Image attached: {path}")
            continue
        if text.startswith("/"):
            print(f"Unknown command: {text}")
            continue
        if not text and not pending_images:
            continue

        try:
            result = agent.chat_sync(text, pending_images)
        except Exception as exc:  # keep the session alive on API errors
            print(f"Error: {exc}", file=sys.stderr)
            continue
        pending_images.clear()
        print(f"agent> {result.output}")


def main() -> None:
    args = _parse_args()
    settings = Settings()
    if args.model:
        settings.model = args.model

    try:
        agent = RobotAgent(settings=settings, tools=[capture_photo])
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.prompt:
        try:
            result = agent.chat_sync(" ".join(args.prompt), args.image)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(result.output)
        return

    _run_repl(agent, list(args.image))


if __name__ == "__main__":
    main()
