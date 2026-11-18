"""Configuration loaded from environment variables and an optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the agent.

    Values come from environment variables or a ``.env`` file in the working
    directory. Variable names match the field names uppercased: ``API_KEY``,
    ``BASE_URL``, ``MODEL``, ``INSTRUCTIONS``. Any OpenAI-compatible Responses
    API endpoint works — the defaults below just point at DeepSeek.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    instructions: str = (
        "You are a helpful assistant running inside the RobotAgent framework. "
        "You can only use tools explicitly provided in the current request. "
        "If no tools are provided, say plainly that you have no tools — never invent or "
        "claim capabilities you don't have. Do not claim to be any other product or assistant."
    )
