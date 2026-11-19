"""Model wrapper that prunes old camera images from the request history.

Each agent turn re-sends the whole conversation, including every camera frame
returned so far; with ~10-20 frames per episode the per-request payload (and
latency) grows quadratically. Only the most recent frames are useful for
control, so older image parts are replaced by a short text placeholder before
the request is sent. This cuts per-request tokens/latency substantially
without changing what the model can perceive (it always sees the latest views).
"""

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel

PLACEHOLDER = "[older camera frame omitted to save bandwidth; request a fresh view if needed]"


class ImagePruningModel(WrapperModel):
    """Keep only the ``keep_images`` most recent image parts in the history."""

    def __init__(self, wrapped, keep_images: int = 2):
        super().__init__(wrapped)
        self.keep_images = keep_images

    def prepare_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
    ) -> list[ModelMessage]:
        messages = super().prepare_messages(messages, model_request_parameters)
        # Count images from the newest backwards; mark older ones for removal.
        drop: set[int] = set()
        seen = 0
        for msg in reversed(messages):
            if not isinstance(msg, ModelRequest):
                continue
            for part in msg.parts:
                content = getattr(part, "content", None)
                items = content if isinstance(content, list) else [content]
                for item in items:
                    if isinstance(item, BinaryContent) and item.media_type.startswith("image/"):
                        seen += 1
                        if seen > self.keep_images:
                            drop.add(id(item))
        if not drop:
            return messages
        pruned: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, ModelRequest):
                pruned.append(msg)
                continue
            new_parts = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                    new_content = [
                        PLACEHOLDER if isinstance(c, BinaryContent) and id(c) in drop else c
                        for c in part.content
                    ]
                    new_parts.append(UserPromptPart(content=new_content))
                elif isinstance(part, ToolReturnPart) and isinstance(part.content, BinaryContent):
                    if id(part.content) in drop:
                        new_parts.append(
                            ToolReturnPart(tool_name=part.tool_name, content=PLACEHOLDER, tool_call_id=part.tool_call_id)
                        )
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            pruned.append(ModelRequest(parts=new_parts))
        return pruned

class TransientRetryModel(WrapperModel):
    """Retry transient gateway rejections (RPM throttle / thinking-passback /
    5xx) with exponential backoff so a campaign is not killed by provider
    instability. Non-transient errors propagate immediately."""

    TRANSIENT_MARKERS = ("rpm", "thinking", "rate", "timeout", "temporarily", "overloaded")

    def __init__(self, wrapped, attempts: int = 4):
        super().__init__(wrapped)
        self.attempts = attempts

    async def request(self, messages, model_settings, model_request_parameters):
        import anyio

        last = None
        for i in range(self.attempts):
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                text = str(exc).lower()
                transient = any(m in text for m in self.TRANSIENT_MARKERS) or any(
                    f"status_code: {c}" in text for c in (429, 500, 502, 503, 504)
                )
                if not transient or i == self.attempts - 1:
                    raise
                last = exc
                await anyio.sleep(5 * (2**i))
        raise last  # pragma: no cover
