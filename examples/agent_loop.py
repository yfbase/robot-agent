"""Example: the agent loop, a tool whitelist, and stop conditions.

Run against the real model (needs API_KEY in .env):
    uv run python examples/agent_loop.py

Offline demo (no key needed, uses pydantic-ai's TestModel):
    uv run python examples/agent_loop.py --demo
"""

import argparse

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from robot_agent import RobotAgent


# Stop condition: the loop ends when the model produces a VALID TripAdvice.
class TripAdvice(BaseModel):
    city: str
    summary: str
    bring_umbrella: bool


# Tool whitelist: the model can call ONLY these two functions.
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    fake_db = {"Beijing": "sunny, 25C", "Shanghai": "rainy, 20C"}
    return fake_db.get(city, f"no data for {city}")


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run offline with TestModel")
    args = parser.parse_args()

    agent = RobotAgent(
        model=TestModel() if args.demo else None,
        tools=[get_weather, celsius_to_fahrenheit],  # 1. whitelist
        output_type=TripAdvice,  # 2. natural stop: valid structured output
        end_strategy="early",  # 4. stop as soon as the final output appears
    )

    # 3. hard safety net: at most 5 model requests and 8 tool calls per run.
    limits = UsageLimits(request_limit=5, tool_calls_limit=8)

    # Per-run options (usage_limits, deps, ...) go through the underlying agent.
    result = agent.agent.run_sync(
        "I'm visiting Shanghai tomorrow. Check the weather, convert 20C to "
        "Fahrenheit, then give me trip advice.",
        usage_limits=limits,
    )

    print("Final output:", result.output)
    print("Messages exchanged:", len(result.all_messages()))


if __name__ == "__main__":
    main()
