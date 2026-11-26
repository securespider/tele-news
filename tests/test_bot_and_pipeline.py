"""Tests for authorization, Telegram HTML formatting/chunking, and end-to-end pipeline."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telegram import LinkPreviewOptions
from telegram.constants import ChatType, ParseMode

import bot as bot_module
from bot import (
    build_application,
    chunk_digest_messages,
    cmd_discover,
    cmd_feeds,
    cmd_pause,
    cmd_resume,
    cmd_start,
    cmd_status,
    cmd_topics,
    deliver_digest,
    digest_job,
    format_article_html,
    handle_poll_answer,
    is_authorized,
    is_within_quiet_hours,
    retention_job,
    run_scrape_cycle,
    scrape_job,
)
from config import load_settings
from db import Database
from learning import CORE_TOPICS
from llm import Stage1Item, Stage2Summary


def _make_settings(tmp_path: Path, allowed_chat_id: str = "") -> Any:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                f"TELEGRAM_ALLOWED_CHAT_ID={allowed_chat_id}",
                "LLM_API_KEY=sk-test-key",
                "MAX_ARTICLES_PER_DIGEST=5",
                "MIN_RELEVANCE_SCORE=0.65",
                "DISCOVERY_POLL_EVERY_N_DIGESTS=2",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file)


def _fake_update(
    *,
    chat_id: int | None = None,
    chat_type: str = ChatType.PRIVATE,
    text: str = "/status",
    poll_answer_user_id: int | None = None,
    has_poll_answer: bool = False,
) -> Any:
    if has_poll_answer:
        user = SimpleNamespace(id=poll_answer_user_id) if poll_answer_user_id is not None else None
        return SimpleNamespace(
            poll_answer=SimpleNamespace(poll_id="p1", user=user, option_ids=[0]),
            effective_chat=None,
            effective_message=None,
        )
    return SimpleNamespace(
        poll_answer=None,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_message=SimpleNamespace(text=text),
    )


def test_is_authorized_rules(tmp_path: Path) -> None:
    db = Database(tmp_path / "auth.db")
    db.init_schema()

    # Auto-bind mode: only private /start binds when unset
    settings_auto = _make_settings(tmp_path, allowed_chat_id="")
    assert is_authorized(_fake_update(chat_id=111, text="/digest"), settings_auto, db) is False
    assert (
        is_authorized(
            _fake_update(chat_id=111, chat_type=ChatType.GROUP, text="/start"),
            settings_auto,
            db,
        )
        is False
    )
    assert is_authorized(_fake_update(chat_id=111, text="/start"), settings_auto, db) is True
    assert db.get_state("active_chat_id") == "111"
    assert is_authorized(_fake_update(chat_id=111, text="/digest"), settings_auto, db) is True
    assert is_authorized(_fake_update(chat_id=222, text="/start"), settings_auto, db) is False

    # PollAnswer authorization
    assert is_authorized(_fake_update(has_poll_answer=True, poll_answer_user_id=None), settings_auto, db) is False
    assert is_authorized(_fake_update(has_poll_answer=True, poll_answer_user_id=222), settings_auto, db) is False
    assert is_authorized(_fake_update(has_poll_answer=True, poll_answer_user_id=111), settings_auto, db) is True

    # Explicit allowed_chat_id overrides dynamic binding
    settings_locked = _make_settings(tmp_path, allowed_chat_id="999")
    assert is_authorized(_fake_update(chat_id=111, text="/start"), settings_locked, db) is False
    assert is_authorized(_fake_update(chat_id=999, text="/digest"), settings_locked, db) is True
    assert is_authorized(_fake_update(has_poll_answer=True, poll_answer_user_id=999), settings_locked, db) is True


def test_quiet_hours_overnight_and_daytime() -> None:
    assert is_within_quiet_hours(None) is False
    assert is_within_quiet_hours("23-07", datetime(2026, 11, 22, 23, 30)) is True
    assert is_within_quiet_hours("23-07", datetime(2026, 11, 23, 2, 15)) is True
    assert is_within_quiet_hours("23-07", datetime(2026, 11, 23, 7, 0)) is False
    assert is_within_quiet_hours("23-07", datetime(2026, 11, 23, 14, 0)) is False
    assert is_within_quiet_hours("09-17", datetime(2026, 11, 23, 10, 0)) is True
    assert is_within_quiet_hours("09-17", datetime(2026, 11, 23, 18, 0)) is False


def test_format_article_html_escaping_and_chunking() -> None:
    article = {
        "title": "Test <script> & \"Quotes\"",
        "url": 'https://example.com/q?a=1&b="2"',
        "comments_url": "https://news.ycombinator.com/item?id=123&x=1",
        "relevance_score": 0.91,
    }
    summary = Stage2Summary(
        summary_bullets=["Bullet with <tag> & details", "Second bullet " + ("x" * 2500)],
        why_relevant="Fits <Agents> interest.",
        hn_pulse="Positive <discussion>.",
        poll_label="Test Label",
        topics=["AI Agents & Tool Use", "MCP"],
    )
    rendered = format_article_html(1, article, summary)
    assert len(rendered) <= 3500
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&amp;" in rendered

    blocks = ["B" * 1600, "C" * 1600, "D" * 1600]
    chunks = chunk_digest_messages("<b>Header</b>", blocks, max_len=3800)
    assert len(chunks) == 2
    assert all(len(c) <= 3800 for c in chunks)


class _FakeBot:
    def __init__(self, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_polls: list[dict[str, Any]] = []
        self._seq = 0

    async def send_message(self, **kwargs: Any) -> Any:
        if self.fail_send:
            raise RuntimeError("Telegram network failure")
        self.sent_messages.append(kwargs)
        self._seq += 1
        return SimpleNamespace(message_id=self._seq)

    async def send_poll(self, **kwargs: Any) -> Any:
        self.sent_polls.append(kwargs)
        self._seq += 1
        return SimpleNamespace(
            message_id=self._seq,
            poll=SimpleNamespace(id=f"tg_poll_{self._seq}"),
        )


class _FakeLLM:
    def __init__(self) -> None:
        self.profile_updates = 0

    async def score_candidates(self, candidates: list[dict[str, Any]], _profile: Any) -> dict[str, Stage1Item]:
        return {
            c["id"]: Stage1Item(
                id=c["id"],
                is_ai=True,
                llm_score=0.88,
                topics=["AI Agents & Tool Use", "MCP"],
                reason="Strong AI agent release.",
            )
            for c in candidates
        }

    async def summarize_article(self, article: dict[str, Any], _profile: Any) -> Stage2Summary:
        return Stage2Summary(
            summary_bullets=["Fact 1 with 2x speedup.", "Fact 2 open source weights."],
            why_relevant="Matches your agent tooling focus.",
            hn_pulse="HN likes the latency benchmark." if article.get("hn_comments") else None,
            poll_label=f"{article['title']} (Agents)",
            topics=["AI Agents & Tool Use", "MCP"],
        )

    async def update_preference_profile(self, **kwargs: Any) -> str:
        self.profile_updates += 1
        return "Updated profile favoring AI Agents & Tool Use and MCP tooling."

    async def propose_discovery_topics(self, recent_titles: list[str], existing_topics: list[str]) -> list[str]:
        return ["Test-Time Compute", "Browser-Use Agents"]


@pytest.mark.asyncio
async def test_end_to_end_pipeline_and_poll_idempotency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path, allowed_chat_id="555")
    db = Database(tmp_path / "e2e.db")
    db.init_schema()
    db.seed_feeds(["https://hnrss.org/frontpage"])
    db.seed_core_topics(CORE_TOPICS)
    llm = _FakeLLM()

    now_iso = datetime.now().astimezone().isoformat()

    async def fake_fetch_rss(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        return (
            [
                {
                    "id": "art_e2e_1",
                    "url": "https://example.org/mcp-agent",
                    "comments_url": "https://news.ycombinator.com/item?id=4242",
                    "title": "Open-Source MCP Agent Framework",
                    "source_feed": "https://hnrss.org/frontpage",
                    "published_at": now_iso,
                    "rss_snippet": "New open-source MCP runtime.",
                }
            ],
            {"https://hnrss.org/frontpage": {"ok": True, "etag": '"e1"', "last_modified": None, "title": "HN"}},
        )

    async def fake_enrich(articles: list[dict[str, Any]], max_concurrency: int = 3) -> dict[str, tuple[str | None, str | None]]:
        return {a["id"]: ("Full webpage text with benchmarks.", "- @alice: Fast!") for a in articles}

    monkeypatch.setattr(bot_module, "fetch_new_rss_entries", fake_fetch_rss)
    monkeypatch.setattr(bot_module, "enrich_articles_batch", fake_enrich)

    scrape_summary = await run_scrape_cycle(db, llm, settings)  # type: ignore[arg-type]
    assert scrape_summary["inserted"] == 1
    assert scrape_summary["queued"] == 1

    # Failure in Telegram send_message keeps article queued (never lost)
    failing_bot = _FakeBot(fail_send=True)
    with pytest.raises(RuntimeError, match="Telegram network failure"):
        await deliver_digest(failing_bot, 555, db, llm, settings)  # type: ignore[arg-type]
    assert db.get_article("art_e2e_1")["status"] == "queued"

    # Successful delivery marks delivered, disables previews, and sends feedback poll
    bot = _FakeBot()
    result = await deliver_digest(bot, 555, db, llm, settings)  # type: ignore[arg-type]
    assert result["delivered"] == 1
    assert result["poll_sent"] is True
    assert db.get_article("art_e2e_1")["status"] == "delivered"

    assert len(bot.sent_messages) == 1
    assert bot.sent_messages[0]["parse_mode"] == ParseMode.HTML
    assert isinstance(bot.sent_messages[0]["link_preview_options"], LinkPreviewOptions)
    assert bot.sent_messages[0]["link_preview_options"].is_disabled is True

    assert len(bot.sent_polls) == 1
    poll_call = bot.sent_polls[0]
    assert poll_call["is_anonymous"] is False
    assert poll_call["allows_multiple_answers"] is True

    # Simulate PollAnswer handling via handler + background task completion
    poll_id = db.get_poll("tg_poll_2")["poll_id"]
    tasks: set[asyncio.Task[Any]] = set()
    app = SimpleNamespace(
        bot_data={
            "settings": settings,
            "db": db,
            "llm": llm,
            "pipeline_lock": asyncio.Lock(),
            "profile_lock": asyncio.Lock(),
            "background_tasks": tasks,
        }
    )
    ctx = SimpleNamespace(application=app, bot=bot)
    poll_update = SimpleNamespace(
        poll_answer=SimpleNamespace(poll_id=poll_id, user=SimpleNamespace(id=555), option_ids=[0]),
        effective_chat=None,
        effective_message=None,
    )

    await handle_poll_answer(poll_update, ctx)  # type: ignore[arg-type]
    if tasks:
        await asyncio.gather(*list(tasks))

    assert llm.profile_updates == 1
    assert "MCP tooling" in (db.get_state("preference_summary") or "")
    weights = {r["topic"]: r for r in db.get_all_topic_weights()}
    assert weights["MCP"]["weight"] == 1.25

    # Duplicate PollAnswer update is ignored idempotently
    await handle_poll_answer(poll_update, ctx)  # type: ignore[arg-type]
    assert llm.profile_updates == 1
    assert {r["topic"]: r for r in db.get_all_topic_weights()}["MCP"]["weight"] == 1.25


class _FakeReplyMessage:
    def __init__(self, text: str, replies: list[tuple[str, dict[str, Any]]]) -> None:
        self.text = text
        self._replies = replies

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self._replies.append((text, kwargs))


@pytest.mark.asyncio
async def test_command_handlers_and_digest_job_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path, allowed_chat_id="")
    db = Database(tmp_path / "cmds.db")
    db.init_schema()
    db.seed_core_topics(CORE_TOPICS)
    llm = _FakeLLM()
    bot = _FakeBot()
    replies: list[tuple[str, dict[str, Any]]] = []

    app = SimpleNamespace(
        bot_data={
            "settings": settings,
            "db": db,
            "llm": llm,
            "pipeline_lock": asyncio.Lock(),
            "profile_lock": asyncio.Lock(),
            "background_tasks": set(),
        }
    )

    def make_update(text: str, chat_id: int = 777) -> Any:
        return SimpleNamespace(
            poll_answer=None,
            effective_chat=SimpleNamespace(id=chat_id, type=ChatType.PRIVATE),
            effective_message=_FakeReplyMessage(text, replies),
        )

    def make_ctx(args: list[str] | None = None) -> Any:
        return SimpleNamespace(application=app, bot=bot, args=args or [])

    # /start binds private chat and sends onboarding poll with 8 options
    await cmd_start(make_update("/start"), make_ctx())  # type: ignore[arg-type]
    assert db.get_state("active_chat_id") == "777"
    assert len(replies) == 1
    assert "Welcome to DailyAI" in replies[0][0]
    assert len(bot.sent_polls) == 1
    assert len(bot.sent_polls[0]["options"]) == 8

    # /feeds empty, add unsafe (rejected), add non-RSS (rejected), add valid, list, remove by 1-based index, remove by URL
    await cmd_feeds(make_update("/feeds"), make_ctx())  # type: ignore[arg-type]
    assert "No enabled feeds" in replies[-1][0]

    await cmd_feeds(make_update("/feeds add"), make_ctx(["add", "http://127.0.0.1/rss"]))  # type: ignore[arg-type]
    assert "unsafe" in replies[-1][0].lower()

    async def fake_validate_feed(url: str) -> tuple[bool, str | None, str | None]:
        if "news.ycombinator.com" in url:
            return (False, None, "URL did not return a valid RSS or Atom feed.")
        return (True, "Mock Feed", None)

    monkeypatch.setattr(bot_module, "validate_rss_feed_url", fake_validate_feed)

    await cmd_feeds(make_update("/feeds add"), make_ctx(["add", "https://news.ycombinator.com"]))  # type: ignore[arg-type]
    assert "rejected" in replies[-1][0].lower()
    assert db.get_all_feeds() == []

    await cmd_feeds(make_update("/feeds add"), make_ctx(["add", "https://simonwillison.net/atom/everything/"]))  # type: ignore[arg-type]
    assert "added/enabled" in replies[-1][0]
    await cmd_feeds(make_update("/feeds add"), make_ctx(["add", "https://hnrss.org/frontpage"]))  # type: ignore[arg-type]

    await cmd_feeds(make_update("/feeds"), make_ctx())  # type: ignore[arg-type]
    assert "1." in replies[-1][0] and "2." in replies[-1][0]

    await cmd_feeds(make_update("/feeds remove 1"), make_ctx(["remove", "1"]))  # type: ignore[arg-type]
    assert "disabled" in replies[-1][0]
    assert [f["url"] for f in db.get_enabled_feeds()] == ["https://hnrss.org/frontpage"]

    await cmd_feeds(make_update("/feeds remove"), make_ctx(["remove", "https://hnrss.org/frontpage"]))  # type: ignore[arg-type]
    assert db.get_enabled_feeds() == []

    # /topics, /status, /pause, /resume
    await cmd_topics(make_update("/topics"), make_ctx())  # type: ignore[arg-type]
    assert "Your AI Interest Profile" in replies[-1][0]

    await cmd_pause(make_update("/pause"), make_ctx())  # type: ignore[arg-type]
    assert db.get_state("digests_paused") == "1"

    await cmd_status(make_update("/status"), make_ctx())  # type: ignore[arg-type]
    assert "Paused" in replies[-1][0]

    # digest_job skips when paused
    called_deliver = False

    async def fake_deliver(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called_deliver
        called_deliver = True
        return {"delivered": 1}

    monkeypatch.setattr(bot_module, "deliver_digest", fake_deliver)
    await digest_job(make_ctx())  # type: ignore[arg-type]
    assert called_deliver is False

    await cmd_resume(make_update("/resume"), make_ctx())  # type: ignore[arg-type]
    assert db.get_state("digests_paused") == "0"

    # digest_job skips inside quiet hours
    app.bot_data["settings"] = settings.model_copy(update={"quiet_hours": "00-23"})
    monkeypatch.setattr(bot_module, "is_within_quiet_hours", lambda qh: True)
    await digest_job(make_ctx())  # type: ignore[arg-type]
    assert called_deliver is False

    # /discover handles empty DB gracefully, then sends discovery poll once AI titles exist
    await cmd_discover(make_update("/discover"), make_ctx())  # type: ignore[arg-type]
    assert "Run /digest first" in replies[-1][0]

    db.insert_new_articles(
        [
            {
                "id": "disc_art_1",
                "url": "https://example.com/disc",
                "comments_url": None,
                "title": "Test-Time Compute Scaling Results",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": "2026-11-22T10:00:00+00:00",
                "rss_snippet": "Scaling reasoning.",
            }
        ]
    )
    db.update_article_stage1_score("disc_art_1", 0.9, 0.9, "AI", ["LLMs & Foundation Models"], "queued")

    await cmd_discover(make_update("/discover"), make_ctx())  # type: ignore[arg-type]
    assert len(bot.sent_polls) == 2
    assert bot.sent_polls[1]["options"] == ["Test-Time Compute", "Browser-Use Agents", "None of these"]


@pytest.mark.asyncio
async def test_run_scrape_cycle_per_article_error_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path, allowed_chat_id="555")
    db = Database(tmp_path / "isolation.db")
    db.init_schema()
    db.seed_feeds(["https://hnrss.org/frontpage"])
    db.seed_core_topics(CORE_TOPICS)

    now_iso = datetime.now().astimezone().isoformat()

    async def fake_fetch_rss(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        return (
            [
                {
                    "id": "art_1",
                    "url": "https://example.org/1",
                    "comments_url": None,
                    "title": "AI Story 1",
                    "source_feed": "https://hnrss.org/frontpage",
                    "published_at": now_iso,
                    "rss_snippet": "AI agent runtime.",
                },
                {
                    "id": "art_2",
                    "url": "https://example.org/2",
                    "comments_url": None,
                    "title": "Failing Story 2",
                    "source_feed": "https://hnrss.org/frontpage",
                    "published_at": now_iso,
                    "rss_snippet": "Transient timeout.",
                },
                {
                    "id": "art_3",
                    "url": "https://example.org/3",
                    "comments_url": None,
                    "title": "Non-AI Gardening Story 3",
                    "source_feed": "https://hnrss.org/frontpage",
                    "published_at": now_iso,
                    "rss_snippet": "Tomatoes.",
                },
            ],
            {"https://hnrss.org/frontpage": {"ok": True, "etag": None, "last_modified": None, "title": "HN"}},
        )

    class _PartialFailLLM:
        def __init__(self) -> None:
            self.called_ids: list[str] = []

        async def score_candidate(self, candidate: dict[str, Any], _profile: Any) -> Stage1Item:
            self.called_ids.append(candidate["id"])
            if candidate["id"] == "art_2":
                raise RuntimeError("Single article LLM failure")
            if candidate["id"] == "art_1":
                return Stage1Item(
                    id="art_1",
                    is_ai=True,
                    llm_score=0.9,
                    topics=["AI Agents & Tool Use"],
                    reason="Strong AI story.",
                )
            return Stage1Item(
                id="art_3",
                is_ai=False,
                llm_score=0.1,
                topics=[],
                reason="Not AI.",
            )

    monkeypatch.setattr(bot_module, "fetch_new_rss_entries", fake_fetch_rss)
    llm = _PartialFailLLM()

    summary = await run_scrape_cycle(db, llm, settings)  # type: ignore[arg-type]
    assert set(llm.called_ids) == {"art_1", "art_2", "art_3"}
    assert summary["inserted"] == 3
    assert summary["scored"] == 2
    assert summary["queued"] == 1
    assert summary["errors"] == 1

    assert db.get_article("art_1")["status"] == "queued"
    assert db.get_article("art_3")["status"] == "filtered_out"
    art2 = db.get_article("art_2")
    assert art2["status"] == "new"
    assert art2["stage1_attempts"] == 1


@pytest.mark.asyncio
async def test_scrape_job_auto_disables_failing_feed_and_alerts_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "TELEGRAM_ALLOWED_CHAT_ID=555",
                "LLM_API_KEY=sk-test-key",
                "FEED_MAX_CONSECUTIVE_FAILURES=2",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    db = Database(tmp_path / "autodelete.db")
    db.init_schema()
    db.seed_feeds(["https://dead.example/rss", "https://healthy.example/rss"])
    db.seed_core_topics(CORE_TOPICS)
    llm = _FakeLLM()
    bot = _FakeBot()

    async def fake_fetch_rss(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        feeds_arg = args[0]
        updates: dict[str, dict[str, Any]] = {}
        for feed in feeds_arg:
            if feed["url"] == "https://dead.example/rss":
                updates[feed["url"]] = {"ok": False, "error": "Response is not a valid RSS or Atom feed"}
            else:
                updates[feed["url"]] = {"ok": True, "etag": None, "last_modified": None, "title": "Healthy"}
        return ([], updates)

    monkeypatch.setattr(bot_module, "fetch_new_rss_entries", fake_fetch_rss)

    app = SimpleNamespace(
        bot_data={
            "settings": settings,
            "db": db,
            "llm": llm,
            "pipeline_lock": asyncio.Lock(),
            "profile_lock": asyncio.Lock(),
            "background_tasks": set(),
        }
    )
    ctx = SimpleNamespace(application=app, bot=bot)

    # First failure: warning recorded, feed still enabled, no alert yet
    await scrape_job(ctx)  # type: ignore[arg-type]
    assert len(db.get_enabled_feeds()) == 2
    assert bot.sent_messages == []

    # Second failure: threshold=2 reached -> soft-disabled & Telegram alert sent
    await scrape_job(ctx)  # type: ignore[arg-type]
    assert [f["url"] for f in db.get_enabled_feeds()] == ["https://healthy.example/rss"]
    assert len(bot.sent_messages) == 1
    alert_text = bot.sent_messages[0]["text"]
    assert "https://dead.example/rss" in alert_text
    assert "/feeds add" in alert_text


@pytest.mark.asyncio
async def test_retention_job_and_job_queue_registration(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "LLM_API_KEY=sk-test-key",
                "ARTICLE_RETENTION_DAYS=10",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    db = Database(tmp_path / "retention.db")
    db.init_schema()
    llm = _FakeLLM()

    old_ts = "2020-01-01T00:00:00+00:00"
    with db._connect() as conn:
        for art_id, status in [("old_exp", "expired"), ("old_deliv", "delivered")]:
            conn.execute(
                """
                INSERT INTO articles (id, url, title, source_feed, published_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (art_id, f"https://example.com/{art_id}", art_id, "https://hnrss.org/frontpage", old_ts, status, old_ts),
            )

    app_data = SimpleNamespace(
        bot_data={
            "settings": settings,
            "db": db,
            "llm": llm,
            "pipeline_lock": asyncio.Lock(),
            "profile_lock": asyncio.Lock(),
            "background_tasks": set(),
        }
    )
    await retention_job(SimpleNamespace(application=app_data))  # type: ignore[arg-type]
    assert db.get_article("old_exp") is None
    assert db.get_article("old_deliv") is not None
    assert db.get_state("last_prune_at") is not None

    built_app = build_application(settings, db, llm)  # type: ignore[arg-type]
    assert built_app.job_queue is not None
    assert [j.name for j in built_app.job_queue.jobs() if j.name == "retention_job"] == ["retention_job"]

    disabled_settings = settings.model_copy(update={"article_retention_days": 0})
    disabled_app = build_application(disabled_settings, db, llm)  # type: ignore[arg-type]
    assert disabled_app.job_queue is not None
    assert [j.name for j in disabled_app.job_queue.jobs() if j.name == "retention_job"] == []

