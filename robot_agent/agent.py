"""The RobotAgent wrapper: a thin layer over pydantic_ai.Agent.

It adds three things on top of pydantic-ai: config-driven model creation,
image-input convenience, and automatic conversation-history management.
Everything else (tools, structured output, streaming, ...) stays available
through the underlying ``agent`` attribute.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .inputs import ImageInput, load_image


@dataclass
class ChatResult:
    """One completed turn."""

    output: str
    messages: list[ModelMessage]
    """Full conversation history including this turn; pass it back to
    pydantic-ai if you ever need manual control."""


class RobotAgent:
    """A conversational agent that accepts text and images.

    Args:
        model: A pre-built pydantic-ai model. When omitted, an OpenAI-compatible
            Responses API model is created from ``settings``.
        settings: Runtime settings; loaded from the environment when omitted.
        instructions: System instructions; defaults to ``settings.instructions``.
        **agent_kwargs: Forwarded to ``pydantic_ai.Agent`` (e.g. ``tools=...``).
    """

    def __init__(
        self,
        model: Model | None = None,
        settings: Settings | None = None,
        instructions: str | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self.settings = settings or Settings()
        if model is None:
            if not self.settings.api_key:
                raise RuntimeError(
                    "No API key configured. Set API_KEY in your .env file or in the environment."
                )
            model = OpenAIResponsesModel(
                self.settings.model,
                provider=OpenAIProvider(
                    base_url=self.settings.base_url, api_key=self.settings.api_key
                ),
            )
        self.model = model
        self.agent = Agent(
            self.model,
            instructions=instructions or self.settings.instructions,
            **agent_kwargs,
        )
        self.history: list[ModelMessage] = []

    def _build_prompt(self, text: str, images: Sequence[ImageInput]) -> Any:
        if not text and not images:
            raise ValueError("Nothing to send: provide text or at least one image.")
        if not images:
            return text
        parts: list[Any] = [text] if text else []
        parts.extend(load_image(image) for image in images)
        return parts

    async def chat(self, text: str = "", images: Sequence[ImageInput] = ()) -> ChatResult:
        """Send a turn (async) and remember it in the conversation history."""
        result = await self.agent.run(
            self._build_prompt(text, images), message_history=self.history
        )
        self.history = list(result.all_messages())
        return ChatResult(output=result.output, messages=self.history)

    def chat_sync(self, text: str = "", images: Sequence[ImageInput] = ()) -> ChatResult:
        """Send a turn (blocking) and remember it in the conversation history."""
        result = self.agent.run_sync(self._build_prompt(text, images), message_history=self.history)
        self.history = list(result.all_messages())
        return ChatResult(output=result.output, messages=self.history)

    def clear_history(self) -> None:
        """Forget the conversation so far."""
        self.history = []
