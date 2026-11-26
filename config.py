"""Configuration loader and validator for the DailyAI Telegram agent."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator, model_validator

AGENT_DIR = Path(__file__).resolve().parent

PLACEHOLDER_KEYS = {
    "",
    "your_llm_api_key_here",
    "REPLACE_WITH_YOUR_LLM_API_KEY",
    "your_telegram_bot_token_here",
}

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_LLM_PROVIDERS = {"openai", "codex"}
VALID_CODEX_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
VALID_DIGEST_MODES = {"balanced", "papers", "news"}
DEFAULT_CODEX_CLI_FALLBACK = (
    "/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex"
)
QUIET_HOURS_RE = re.compile(r"^([01]?\d|2[0-3])-([01]?\d|2[0-3])$")

DEFAULT_RSS_FEEDS = [
    "https://hnrss.org/frontpage",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://simonwillison.net/atom/everything/",
    "https://huggingface.co/blog/feed.xml",
    "https://openai.com/news/rss.xml",
    "https://rss.arxiv.org/rss/cs.AI+cs.LG+cs.CL",
]

DEFAULT_ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV"]


class Settings(BaseModel):
    """Typed, validated configuration for the DailyAI Telegram agent."""

    telegram_bot_token: str
    telegram_allowed_chat_id: int | None = None
    org_name: str = "DailyAI Research"
    telegram_admin_ids: list[int] = Field(default_factory=list)
    telegram_allowed_user_ids: list[int] = Field(default_factory=list)
    allow_self_register_first_admin: bool = True

    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=45.0, ge=5.0)
    llm_max_concurrency: int = Field(default=4, ge=1, le=32)
    codex_cli_path: str | None = None
    codex_reasoning_effort: str = "low"

    rss_feeds: list[str] = Field(default_factory=lambda: list(DEFAULT_RSS_FEEDS))
    enable_hf_daily_papers: bool = True
    enable_arxiv_api: bool = True
    arxiv_categories: list[str] = Field(default_factory=lambda: list(DEFAULT_ARXIV_CATEGORIES))
    semantic_scholar_api_key: str = ""

    scrape_interval_minutes: int = Field(default=30, ge=1)
    digest_interval_hours: float = Field(default=4.0, ge=0.05)
    digest_tick_minutes: int = Field(default=15, ge=1, le=240)
    max_articles_per_digest: int = 5
    min_relevance_score: float = Field(default=0.65, ge=0.0, le=1.0)
    max_article_age_hours: int = Field(default=72, ge=1)
    max_entries_per_feed: int = Field(default=30, ge=1)
    discovery_poll_every_n_digests: int = Field(default=3, ge=1)
    feed_max_consecutive_failures: int = Field(default=5, ge=0)
    article_retention_days: int = Field(default=30, ge=0)
    quiet_hours: str | None = None
    default_digest_mode: str = "balanced"

    rate_limit_commands_per_min: int = Field(default=20, ge=1, le=300)
    rate_limit_llm_per_user_hour: int = Field(default=12, ge=1, le=200)
    daily_org_llm_budget: int = Field(default=500, ge=10, le=50000)

    log_level: str = "INFO"
    db_path: Path = Field(default_factory=lambda: AGENT_DIR / "data" / "daily_ai.db")

    @field_validator("telegram_bot_token", "llm_base_url", "llm_model", "org_name", mode="before")
    @classmethod
    def _strip_required_strings(cls, value: Any) -> str:
        if value is None:
            raise ValueError("Required configuration string cannot be empty")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("Required configuration string cannot be empty")
        return cleaned

    @field_validator("llm_model")
    @classmethod
    def _validate_llm_model_safe(cls, value: str) -> str:
        if value.startswith("-"):
            raise ValueError("LLM_MODEL cannot start with '-'")
        return value

    @field_validator("llm_api_key", "semantic_scholar_api_key", mode="before")
    @classmethod
    def _strip_optional_secret(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _validate_llm_provider(cls, value: Any) -> str:
        text = str(value or "openai").strip().lower()
        if text not in VALID_LLM_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(VALID_LLM_PROVIDERS)}")
        return text

    @field_validator("codex_reasoning_effort", mode="before")
    @classmethod
    def _validate_codex_reasoning_effort(cls, value: Any) -> str:
        text = str(value or "low").strip().lower()
        if text not in VALID_CODEX_REASONING_EFFORTS:
            raise ValueError(
                f"CODEX_REASONING_EFFORT must be one of {sorted(VALID_CODEX_REASONING_EFFORTS)}"
            )
        return text

    @field_validator("default_digest_mode", mode="before")
    @classmethod
    def _validate_digest_mode(cls, value: Any) -> str:
        text = str(value or "balanced").strip().lower()
        if text not in VALID_DIGEST_MODES:
            raise ValueError(f"DEFAULT_DIGEST_MODE must be one of {sorted(VALID_DIGEST_MODES)}")
        return text

    @field_validator("codex_cli_path", mode="before")
    @classmethod
    def _strip_optional_codex_cli_path(cls, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @model_validator(mode="after")
    def _require_api_key_for_openai(self) -> Settings:
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ValueError("LLM_API_KEY cannot be empty when LLM_PROVIDER=openai")
        return self

    @field_validator("telegram_allowed_chat_id", mode="before")
    @classmethod
    def _parse_optional_chat_id(cls, value: Any) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)

    @field_validator("telegram_admin_ids", "telegram_allowed_user_ids", mode="before")
    @classmethod
    def _parse_int_id_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = [str(item) for item in value]
        else:
            raise ValueError("User ID list must be comma-separated integers or a list")

        deduped: list[int] = []
        seen: set[int] = set()
        for item in raw_items:
            cleaned = item.strip()
            if not cleaned:
                continue
            uid = int(cleaned)
            if uid not in seen:
                seen.add(uid)
                deduped.append(uid)
        return deduped

    @field_validator("rss_feeds", mode="before")
    @classmethod
    def _parse_rss_feeds(cls, value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_RSS_FEEDS)
        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = [str(item) for item in value]
        else:
            raise ValueError("RSS_FEEDS must be a comma-separated string or list of URLs")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped

    @field_validator("arxiv_categories", mode="before")
    @classmethod
    def _parse_arxiv_categories(cls, value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_ARXIV_CATEGORIES)
        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple)):
            raw_items = [str(item) for item in value]
        else:
            raise ValueError("ARXIV_CATEGORIES must be a comma-separated string or list")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        return deduped or list(DEFAULT_ARXIV_CATEGORIES)

    @field_validator("max_articles_per_digest", mode="before")
    @classmethod
    def _clamp_max_articles(cls, value: Any) -> int:
        parsed = int(value)
        return max(1, min(9, parsed))

    @field_validator("quiet_hours", mode="before")
    @classmethod
    def _validate_quiet_hours(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = QUIET_HOURS_RE.fullmatch(text)
        if not match:
            raise ValueError("QUIET_HOURS must be in 24-hour HH-HH format (e.g., 23-07)")
        start_hour = int(match.group(1))
        end_hour = int(match.group(2))
        if start_hour == end_hour:
            raise ValueError("QUIET_HOURS start and end hours cannot be identical")
        return f"{start_hour:02d}-{end_hour:02d}"

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: Any) -> str:
        text = str(value or "INFO").strip().upper()
        if text not in VALID_LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")
        return text

    @field_validator("db_path", mode="before")
    @classmethod
    def _resolve_db_path(cls, value: Any) -> Path:
        text = str(value or "").strip()
        if not text:
            return AGENT_DIR / "data" / "daily_ai.db"
        path = Path(text)
        if not path.is_absolute():
            path = AGENT_DIR / path
        return path


def resolve_codex_cli_path(settings: Settings) -> Path | None:
    """Locate the Codex CLI executable from settings, PATH, or default macOS app bundle."""
    if settings.codex_cli_path:
        return Path(settings.codex_cli_path)
    which_path = shutil.which("codex")
    if which_path:
        return Path(which_path)
    fallback = Path(DEFAULT_CODEX_CLI_FALLBACK)
    if fallback.exists():
        return fallback
    return None


def _non_empty(mapping: dict[str, str | None], key: str) -> str | None:
    if key not in mapping:
        return None
    raw = mapping[key]
    if raw is None:
        return ""
    return raw


def load_settings(env_path: Path | None = None) -> Settings:
    """Load Settings from an explicit .env file or AGENT_DIR/.env + os.environ."""
    merged: dict[str, str | None] = {}
    if env_path is not None:
        if env_path.exists():
            merged.update(dotenv_values(env_path))
    else:
        default_env = AGENT_DIR / ".env"
        if default_env.exists():
            merged.update(dotenv_values(default_env))
        for key, val in os.environ.items():
            merged[key] = val

    payload: dict[str, Any] = {
        "telegram_bot_token": _non_empty(merged, "TELEGRAM_BOT_TOKEN"),
        "telegram_allowed_chat_id": _non_empty(merged, "TELEGRAM_ALLOWED_CHAT_ID"),
        "llm_api_key": _non_empty(merged, "LLM_API_KEY") or "",
    }

    optional_env_map = {
        "ORG_NAME": "org_name",
        "TELEGRAM_ADMIN_IDS": "telegram_admin_ids",
        "TELEGRAM_ALLOWED_USER_IDS": "telegram_allowed_user_ids",
        "ALLOW_SELF_REGISTER_FIRST_ADMIN": "allow_self_register_first_admin",
        "LLM_PROVIDER": "llm_provider",
        "LLM_BASE_URL": "llm_base_url",
        "LLM_MODEL": "llm_model",
        "LLM_TIMEOUT_SECONDS": "llm_timeout_seconds",
        "LLM_MAX_CONCURRENCY": "llm_max_concurrency",
        "CODEX_CLI_PATH": "codex_cli_path",
        "CODEX_REASONING_EFFORT": "codex_reasoning_effort",
        "RSS_FEEDS": "rss_feeds",
        "ENABLE_HF_DAILY_PAPERS": "enable_hf_daily_papers",
        "ENABLE_ARXIV_API": "enable_arxiv_api",
        "ARXIV_CATEGORIES": "arxiv_categories",
        "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        "SCRAPE_INTERVAL_MINUTES": "scrape_interval_minutes",
        "DIGEST_INTERVAL_HOURS": "digest_interval_hours",
        "DIGEST_TICK_MINUTES": "digest_tick_minutes",
        "MAX_ARTICLES_PER_DIGEST": "max_articles_per_digest",
        "MIN_RELEVANCE_SCORE": "min_relevance_score",
        "MAX_ARTICLE_AGE_HOURS": "max_article_age_hours",
        "MAX_ENTRIES_PER_FEED": "max_entries_per_feed",
        "DISCOVERY_POLL_EVERY_N_DIGESTS": "discovery_poll_every_n_digests",
        "FEED_MAX_CONSECUTIVE_FAILURES": "feed_max_consecutive_failures",
        "ARTICLE_RETENTION_DAYS": "article_retention_days",
        "QUIET_HOURS": "quiet_hours",
        "DEFAULT_DIGEST_MODE": "default_digest_mode",
        "RATE_LIMIT_COMMANDS_PER_MIN": "rate_limit_commands_per_min",
        "RATE_LIMIT_LLM_PER_USER_HOUR": "rate_limit_llm_per_user_hour",
        "DAILY_ORG_LLM_BUDGET": "daily_org_llm_budget",
        "LOG_LEVEL": "log_level",
        "DB_PATH": "db_path",
    }
    for env_key, field_name in optional_env_map.items():
        val = _non_empty(merged, env_key)
        if val is not None and (
            val.strip() != ""
            or field_name
            in {
                "telegram_allowed_chat_id",
                "quiet_hours",
                "rss_feeds",
                "codex_cli_path",
                "telegram_admin_ids",
                "telegram_allowed_user_ids",
                "semantic_scholar_api_key",
            }
        ):
            payload[field_name] = val

    return Settings(**payload)


def validate_runtime_secrets(settings: Settings) -> None:
    """Ensure required API secrets or Codex CLI runtime are populated and valid."""
    tg_token = settings.telegram_bot_token.strip()
    if tg_token in PLACEHOLDER_KEYS or tg_token.startswith("your_"):
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is missing or still a placeholder. "
            "Set a valid BotFather token in agent/.env."
        )

    if settings.llm_provider == "codex":
        codex_path = resolve_codex_cli_path(settings)
        if codex_path is None or not codex_path.exists() or not os.access(codex_path, os.X_OK):
            raise ValueError(
                "Codex CLI executable not found or not executable. "
                "Set CODEX_CLI_PATH in agent/.env or install the `codex` CLI."
            )
        return

    llm_key = settings.llm_api_key.strip()
    if (
        llm_key in PLACEHOLDER_KEYS
        or llm_key.startswith("your_")
        or llm_key.startswith("REPLACE_WITH_")
    ):
        raise ValueError(
            "LLM_API_KEY is missing or still a placeholder. "
            "Set a valid OpenAI-compatible API key (or use LLM_PROVIDER=codex) in agent/.env."
        )
