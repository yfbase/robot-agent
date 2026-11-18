# RobotAgent

A simple agent framework built on [pydantic-ai](https://ai.pydantic.dev/), managed with
[uv](https://docs.astral.sh/uv/). It accepts **text and image** input and talks to any
**OpenAI-compatible Responses API** endpoint — DeepSeek, Qwen, a local server, etc.
Switching providers is a configuration change, not a code change.

The wrapper is intentionally thin: `RobotAgent` composes `pydantic_ai.Agent` and only adds
config-driven model creation, image-input convenience, and conversation-history management.
Tools, structured output, streaming, and everything else pydantic-ai offers remain available
through the underlying `agent` attribute.

## Quick start

```bash
uv sync                            # installs a managed Python + all dependencies
copy .env.example .env             # then put your API_KEY into .env
uv run python examples/cli.py      # interactive chat
```

Interactive commands:

| Command         | Effect                              |
| --------------- | ----------------------------------- |
| `/image <path>` | attach an image to the next message |
| `/clear`        | forget the conversation history     |
| `/exit`         | quit                                |

One-shot usage:

```bash
uv run python examples/cli.py "hello, who are you?"
uv run python examples/cli.py --image cat.png "what is in this image?"
```

## Library usage

```python
from robot_agent import RobotAgent

agent = RobotAgent()

# Text only
result = agent.chat_sync("Hello!")
print(result.output)

# Text + image (local path, http URL, or raw bytes)
result = agent.chat_sync("What is in this image?", images=["cat.png"])
print(result.output)

# Async works too
result = await agent.chat("And in one word?")
```

History is kept automatically between calls; `agent.clear_history()` resets it.
See [examples/describe_image.py](examples/describe_image.py) for a runnable example.

## Configuration

Settings come from environment variables or a `.env` file (see `.env.example`):

| Variable       | Default                       | Meaning                                        |
| -------------- | ----------------------------- | ---------------------------------------------- |
| `API_KEY`      | — (required)                  | API key for the endpoint                       |
| `BASE_URL`     | `https://api.deepseek.com/v1` | OpenAI-compatible API base URL                 |
| `MODEL`        | `deepseek-v4-flash`           | model name                                     |
| `INSTRUCTIONS` | honest assistant identity (see `config.py`) | system instructions            |

The defaults point at DeepSeek, but nothing in the code is DeepSeek-specific. To use another
OpenAI-compatible service, set `BASE_URL`, `MODEL`, and `API_KEY` — for example a local
server: `BASE_URL=http://localhost:8000/v1`.

Note on images: image support depends on the model. DeepSeek's Flash models accept images
(JPEG, PNG, GIF, WebP); `deepseek-v4-pro` is text-only. Images are sent as standard OpenAI
content parts, so any vision-capable model on a compatible endpoint works.

## Development

```bash
uv run pytest        # offline tests (pydantic-ai TestModel, no key needed)
uv run ruff check .  # lint
uv run ruff format . # format
```

## Project layout

Framework and application code are kept separate: `robot_agent/` is pure library
code; `examples/` holds runnable applications built on top of it.

```
robot_agent/
├── config.py      # pydantic-settings: key, base URL, model, instructions
├── inputs.py      # image loading: path / URL / bytes -> BinaryContent / ImageUrl
├── tools.py       # tools the model can call (camera capture)
└── agent.py       # RobotAgent wrapper (builds the model) + ChatResult

examples/
├── cli.py             # interactive / one-shot command-line chat
├── describe_image.py  # text + image library usage
├── camera_tool.py     # camera tool demo
└── agent_loop.py      # agent loop, tool whitelist, stop conditions
```