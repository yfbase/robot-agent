"""Example: give the model a camera tool and ask it to look around.

Run (needs API_KEY in .env, a connected camera, and a vision model):
    uv run python examples/camera_tool.py
"""

from robot_agent import RobotAgent
from robot_agent.tools import capture_photo


def main() -> None:
    agent = RobotAgent(tools=[capture_photo])
    result = agent.chat_sync("Take a photo with the camera and describe what you see.")
    print(result.output)


if __name__ == "__main__":
    main()
