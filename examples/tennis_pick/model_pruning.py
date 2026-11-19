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