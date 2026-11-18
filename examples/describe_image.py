"""Example: text-only and text+image calls with RobotAgent.

Run:
    uv run python examples/describe_image.py [image_path]

Requires API_KEY in .env or the environment.
"""

import sys

from robot_agent import RobotAgent


def main() -> None:
    agent = RobotAgent()

    result = agent.chat_sync("Hello! In one sentence, what can you do?")
    print("Text-only reply:", result.output)

    if len(sys.argv) > 1:
        result = agent.chat_sync("What is in this image?", images=[sys.argv[1]])
        print("Image reply:", result.output)


if __name__ == "__main__":
    main()
