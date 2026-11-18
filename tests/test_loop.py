"""Offline tests for loop behavior: tool calling, natural stop, and hard limits."""

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from robot_agent import RobotAgent


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "sunny, 25C"


def _parts(messages):
    return [part for message in messages for part in message.parts]


def test_loop_calls_tools_then_stops_at_final_output():
    agent = RobotAgent(model=TestModel(), tools=[get_weather])
    result = agent.agent.run_sync("What's the weather in Beijing?")
    parts = _parts(result.all_messages())
    assert any(isinstance(part, ToolCallPart) and part.tool_name == "get_weather" for part in parts)
    assert any(isinstance(part, ToolReturnPart) for part in parts)
    assert isinstance(result.output, str) and result.output


def test_request_limit_stops_a_runaway_loop():
    # TestModel(call_tools="all") keeps calling tools; request_limit=1 cuts it off.
    agent = RobotAgent(model=TestModel(call_tools="all"), tools=[get_weather])
    with pytest.raises(UsageLimitExceeded):
        agent.agent.run_sync("weather?", usage_limits=UsageLimits(request_limit=1))


def test_unregistered_tools_are_not_available():
    agent = RobotAgent(model=TestModel(call_tools=[]), tools=[get_weather])
    result = agent.agent.run_sync("hello")
    tool_names = {
        part.tool_name for part in _parts(result.all_messages()) if isinstance(part, ToolCallPart)
    }
    assert tool_names <= {"get_weather"}
