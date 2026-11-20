"""Example: give the model real-arm tools and ask it to reset the arm.

Run (needs API_KEY in .env and the SO101 arm connected via its USB serial
adapter):
    uv run python examples/arm_tool.py

The arm holds its pose while the agent runs; when the ``with`` block exits,
all servos are disabled (torque off) so the arm can be moved by hand.
"""

from robot_agent import RobotAgent
from robot_agent.tools import make_arm_tools
from simple_arm import SimpleArm


def main() -> None:
    with SimpleArm() as arm:
        agent = RobotAgent(tools=make_arm_tools(arm))
        result = agent.chat_sync(
            "Reset the arm: raise it to the high pose and open the gripper, "
            "then report the final joint angles."
        )
        print(result.output)


if __name__ == "__main__":
    main()