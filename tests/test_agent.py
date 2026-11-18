"""Offline agent tests using pydantic-ai's TestModel — no API key or network needed."""

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from robot_agent import RobotAgent, Settings


def _user_contents(messages):
    contents = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    if isinstance(part.content, str):
                        contents.append(part.content)
                    else:
                        contents.extend(part.content)
    return contents


def test_text_chat_sync():
    agent = RobotAgent(model=TestModel())
    result = agent.chat_sync("hello")
    assert isinstance(result.output, str) and result.output
    assert len(result.messages) >= 2  # request + response


def test_history_grows_across_turns():
    agent = RobotAgent(model=TestModel())
    first = agent.chat_sync("hello")
    second = agent.chat_sync("how are you?")
    assert len(second.messages) > len(first.messages)
    assert second.messages[: len(first.messages)] == first.messages


def test_image_chat(png_bytes):
    agent = RobotAgent(model=TestModel())
    result = agent.chat_sync("what is this?", images=[png_bytes])
    contents = _user_contents(result.messages)
    assert "what is this?" in contents
    assert any(isinstance(item, BinaryContent) for item in contents)


def test_image_only_chat(png_bytes):
    agent = RobotAgent(model=TestModel())
    result = agent.chat_sync(images=[png_bytes])
    assert any(isinstance(item, BinaryContent) for item in _user_contents(result.messages))


def test_clear_history():
    agent = RobotAgent(model=TestModel())
    agent.chat_sync("hello")
    assert agent.history
    agent.clear_history()
    assert agent.history == []


def test_empty_input_raises():
    agent = RobotAgent(model=TestModel())
    with pytest.raises(ValueError, match="Nothing to send"):
        agent.chat_sync()


async def test_chat_async():
    agent = RobotAgent(model=TestModel())
    result = await agent.chat("hello")
    assert isinstance(result.output, str) and result.output


def test_missing_key_raises():
    settings = Settings(_env_file=None, api_key="")
    with pytest.raises(RuntimeError, match="API key"):
        RobotAgent(settings=settings)


def test_settings_accepts_field_names_in_code():
    settings = Settings(_env_file=None, api_key="sk-test", model="my-model")
    assert settings.api_key == "sk-test"
    assert settings.model == "my-model"
    agent = RobotAgent(settings=settings)
    assert type(agent.model).__name__ == "OpenAIResponsesModel"
