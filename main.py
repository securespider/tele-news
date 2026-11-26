"""CLI entrypoint for the DailyAI multi-user Telegram research & digest agent."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise RuntimeError(
        f"Python 3.11+ is required (found {sys.version_info.major}.{sys.version_info.minor}). "
        "Use agent/.venv/bin/python."
    )

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from telegram import Update

from bot import build_application
from config import load_settings, validate_runtime_secrets
from db import Database
from learning import CORE_TOPICS
from llm import LLMClient
from security import SecretRedactingFilter, harden_file_permissions


def configure_logging(level_name: str) -> None:
    """Configure root logging with secret redaction and mute httpx/httpcore URL token logs."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )
    redactor = SecretRedactingFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redactor)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore2").setLevel(logging.WARNING)


def main() -> None:
    """Initialize config, hardened DB, org users, LLM client, and start Telegram polling."""
    settings = load_settings()
    configure_logging(settings.log_level)
    validate_runtime_secrets(settings)
    harden_file_permissions(AGENT_DIR / ".env")

    db = Database(settings.db_path)
    db.init_schema()
    db.seed_feeds(settings.rss_feeds)
    db.seed_core_topics(CORE_TOPICS)

    admin_ids = list(settings.telegram_admin_ids)
    if settings.telegram_allowed_chat_id is not None and settings.telegram_allowed_chat_id not in admin_ids:
        admin_ids.append(settings.telegram_allowed_chat_id)
    db.seed_users_from_settings(admin_ids, settings.telegram_allowed_user_ids, CORE_TOPICS)

    llm = LLMClient(
        settings,
        budget_checker=lambda: db.get_daily_llm_usage() < settings.daily_org_llm_budget,
    )
    application = build_application(settings, db, llm)
    logging.getLogger(__name__).info(
        "Starting DailyAI Telegram agent (org=%s, provider=%s, model=%s, active_users=%d)",
        settings.org_name,
        settings.llm_provider,
        settings.llm_model,
        db.count_active_users(),
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
