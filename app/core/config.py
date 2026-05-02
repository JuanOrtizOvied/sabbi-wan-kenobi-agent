import os
from dotenv import load_dotenv


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


load_dotenv()

# -----------------------------
# ENV
# -----------------------------
SESSION_DB_URL = os.getenv(
    "SESSION_DB_URL",
    "postgresql+asyncpg://admin:pass@localhost:5432/ai-experiments",
)

CREATE_SESSION_TABLES = _to_bool(os.getenv("CREATE_SESSION_TABLES", "1"), default=True)

SESSIONS_TABLE = os.getenv("AGENT_SESSIONS_TABLE", "agent_sessions")
MESSAGES_TABLE = os.getenv("AGENT_MESSAGES_TABLE", "agent_messages")


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    SABBI_EXPERTO_OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    SABBI_VECTOR_STORE_ID: str = os.getenv("SABBI_VECTOR_STORE_ID", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")


settings = Settings()
