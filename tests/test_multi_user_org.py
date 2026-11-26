"""Tests for 20+ user organization RBAC, invite onboarding, isolated personalization, and shared caching."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from telegram.constants import ChatType

import bot as bot_module
from bot import (
    cmd_admin,
    cmd_digest,
    cmd_feeds,
    cmd_join,
    cmd_settings,
    deliver_digest,
    digest_job,
    handle_callback_query,
    handle_poll_answer,
)
from config import load_settings
from db import Database
from learning import CORE_TOPICS
from llm import Stage2Summary
from security import CallbackSigner


def _make_org_settings(tmp_path: Path) -> Any:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:ORG_TEST_TOKEN",
                "ORG_NAME=Acme AI Lab",
                "TELEGRAM_ADMIN_IDS=1001,1002",
                "LLM_API_KEY=sk-test-org",
                "MIN_RELEVANCE_SCORE=0.60",
                "MAX_ARTICLES_PER_DIGEST=5",
                "DAILY_ORG_LLM_BUDGET=500",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file)


class _OrgFakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_polls: list[dict[str, Any]] = []
        self._seq = 0

    async def send_message(self, **kwargs: Any) -> Any:
        self.sent_messages.append(kwargs)
        self._seq += 1
        return SimpleNamespace(message_id=self._seq)

    async def send_poll(self, **kwargs: Any) -> Any:
        self.sent_polls.append(kwargs)
        self._seq += 1
        return SimpleNamespace(
            message_id=self._seq,
            poll=SimpleNamespace(id=f"org_poll_{self._seq}"),
        )


class _FakeReplyMessage:
    def __init__(self, text: str, replies: list[str]) -> None:
        self.text = text
        self._replies = replies

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self._replies.append(text)


class _CountingLLM:
    def __init__(self) -> None:
        self.summarize_calls = 0
        self.profile_calls = 0

    async def summarize_article(self, article: dict[str, Any], _profile: Any) -> Stage2Summary:
        self.summarize_calls += 1
        return Stage2Summary(
            summary_bullets=[f"Key engineering fact for {article['title']}"],
            why_relevant="High-impact AI engineering release.",
            hn_pulse=None,
            poll_label=article["title"][:60],
            topics=["AI Agents & Tool Use", "LLMs & Foundation Models"],
        )

    async def update_preference_profile(self, **kwargs: Any) -> str:
        self.profile_calls += 1
        return "Personalized org profile summary."

    async def propose_discovery_topics(self, recent_titles: list[str], existing_topics: list[str]) -> list[str]:
        return ["Test-Time Scaling"]


@pytest.mark.asyncio
async def test_25_user_org_rbac_invites_shared_caching_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_org_settings(tmp_path)
    db = Database(tmp_path / "org25.db")
    db.init_schema()
    db.seed_core_topics(CORE_TOPICS)
    db.seed_users_from_settings(settings.telegram_admin_ids, [], CORE_TOPICS)

    llm = _CountingLLM()
    bot = _OrgFakeBot()
    replies: dict[int, list[str]] = {}
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

    def mk(uid: int, text: str, args: list[str] | None = None) -> tuple[Any, Any]:
        replies.setdefault(uid, [])
        upd = SimpleNamespace(
            poll_answer=None,
            effective_user=SimpleNamespace(id=uid, username=f"user{uid}", first_name=f"User{uid}", last_name=""),
            effective_chat=SimpleNamespace(id=uid, type=ChatType.PRIVATE),
            effective_message=_FakeReplyMessage(text, replies[uid]),
        )
        ctx = SimpleNamespace(application=app, bot=bot, args=args or [])
        return upd, ctx

    # 1. Unauthorized user blocked from /digest
    await cmd_digest(*mk(9999, "/digest"))
    assert "requires an invitation" in replies[9999][-1]

    # 2. Admin 1001 creates a 25-use invite code; 23 members (2001..2023) join via /join
    await cmd_admin(*mk(1001, "/admin invite", ["invite", "member", "25", "72"]))
    invite_msg = replies[1001][-1]
    token = invite_msg.split("/join ")[1].split("</code>")[0].strip()

    for member_id in range(2001, 2024):
        await cmd_join(*mk(member_id, f"/join {token}", [token]))
        assert "Welcome to" in replies[member_id][-1]

    assert db.count_active_users() == 25  # 2 admins + 23 members

    # 3. RBAC enforcement: member 2001 cannot run /admin or /feeds add, and cannot demote/revoke admins
    await cmd_admin(*mk(2001, "/admin users", ["users"]))
    assert "Admin privileges are required" in replies[2001][-1]

    await cmd_feeds(*mk(2001, "/feeds add", ["add", "https://example.com/rss"]))
    assert "Only organization admins" in replies[2001][-1]

    # Admin cannot demote/revoke the last remaining admin
    await cmd_admin(*mk(1001, "/admin demote", ["demote", "1002"]))
    await cmd_admin(*mk(1001, "/admin demote", ["demote", "1001"]))
    assert "Cannot demote the last active admin" in replies[1001][-1]

    # Revoke member 2023 and verify immediate access loss
    await cmd_admin(*mk(1001, "/admin revoke", ["revoke", "2023"]))
    await cmd_digest(*mk(2023, "/digest"))
    assert "requires an invitation" in replies[2023][-1]

    # 4. Seed 2 shared stories & deliver personalized digests to all 24 active users.
    #    VERIFY Stage 2 LLM summarization runs ONLY ONCE per story (2 calls total, NOT 48)!
    now_iso = datetime.now(timezone.utc).isoformat()
    db.insert_new_articles(
        [
            {
                "id": "hot_story_1",
                "url": "https://example.org/agents-1",
                "title": "Multi-Agent Systems in Production",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": now_iso,
                "rss_snippet": "Agent orchestration patterns.",
                "content_type": "news",
            },
            {
                "id": "hot_paper_2",
                "url": "https://arxiv.org/abs/2611.77777",
                "title": "Sparse MoE Inference Paper",
                "source_feed": "https://huggingface.co/papers",
                "published_at": now_iso,
                "rss_snippet": "2x inference throughput.",
                "content_type": "paper",
                "arxiv_id": "2611.77777",
            },
        ]
    )
    db.update_article_stage1_score("hot_story_1", 0.90, 0.90, "AI", ["AI Agents & Tool Use"], "queued")
    db.update_article_stage1_score("hot_paper_2", 0.88, 0.88, "AI", ["LLMs & Foundation Models"], "queued")

    monkeypatch.setattr(
        bot_module,
        "enrich_articles_batch",
        lambda articles, max_concurrency=3: asyncio.sleep(
            0, result={a["id"]: ("Extracted text.", None) for a in articles}
        ),
    )

    signer = CallbackSigner(settings.telegram_bot_token)
    for uid in [1001, 1002, *range(2001, 2023)]:
        res = await deliver_digest(
            bot, uid, db, llm, settings, user_id=uid, signer=signer  # type: ignore[arg-type]
        )
        assert res["delivered"] == 2

    # Critical cost/scalability guarantee: 2 stories summarized ONCE for all 24 active org users!
    assert llm.summarize_calls == 2

    # Subsequent digest has 0 unread stories for member 2001
    res_empty = await deliver_digest(bot, 2001, db, llm, settings, user_id=2001)  # type: ignore[arg-type]
    assert res_empty["delivered"] == 0

    # 5. Per-user topic weight isolation: member 2001 votes on poll; member 2002 stays at default 1.0
    poll_2001 = next(
        p for p in bot.sent_polls if p["chat_id"] == 2001 and "valuable" in p["question"]
    )
    # Find saved poll_id for user 2001 in DB
    with db._connect() as conn:
        row = conn.execute(
            "SELECT poll_id FROM polls WHERE user_id = 2001 AND poll_type = 'digest_feedback';"
        ).fetchone()
    assert row is not None
    poll_id_2001 = row["poll_id"]

    tasks: set[asyncio.Task[Any]] = set()
    app.bot_data["background_tasks"] = tasks
    await handle_poll_answer(
        SimpleNamespace(
            poll_answer=SimpleNamespace(poll_id=poll_id_2001, user=SimpleNamespace(id=2001), option_ids=[0]),
            effective_chat=None,
            effective_message=None,
        ),
        SimpleNamespace(application=app, bot=bot),  # type: ignore[arg-type]
    )
    if tasks:
        await asyncio.gather(*list(tasks))

    w_2001 = {r["topic"]: r["weight"] for r in db.get_user_topic_weights(2001)}
    w_2002 = {r["topic"]: r["weight"] for r in db.get_user_topic_weights(2002)}
    assert w_2001["AI Agents & Tool Use"] == 1.25
    assert w_2002["AI Agents & Tool Use"] == 1.0

    # 6. Per-user settings & quiet hours + callback signature cross-user isolation
    await cmd_settings(*mk(2002, "/settings mode papers", ["mode", "papers"]))
    assert db.get_user(2002)["digest_mode"] == "papers"
    assert db.get_user(2001)["digest_mode"] == "balanced"

    # Cross-user signed callback rejected
    cb_alerts: list[str] = []
    bad_sig_for_2002 = signer.sign(2001, "sv", "hot_story_1")
    cb_upd = SimpleNamespace(
        callback_query=SimpleNamespace(
            from_user=SimpleNamespace(id=2002),
            data=bad_sig_for_2002,
            message=None,
            answer=lambda text="", show_alert=False: asyncio.sleep(0, result=cb_alerts.append(text)),
        )
    )
    await handle_callback_query(cb_upd, SimpleNamespace(application=app, bot=bot))  # type: ignore[arg-type]
    assert cb_alerts[-1] == "Invalid or expired action button."

    # Audit log recorded security events
    audit_events = [e["event_type"] for e in db.get_recent_audit_events(limit=50)]
    assert "invite_created" in audit_events
    assert "invite_redeemed" in audit_events
    assert "rbac_denied" in audit_events
    assert "user_revoked" in audit_events
    assert "invalid_callback_signature" in audit_events


def test_legacy_single_user_db_migration_preserves_state_and_weights(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    # Build minimal legacy DB before multi-user tables existed
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE feeds (url TEXT PRIMARY KEY, title TEXT, enabled INTEGER NOT NULL DEFAULT 1, added_at TEXT NOT NULL);
        CREATE TABLE articles (id TEXT PRIMARY KEY, url TEXT NOT NULL, comments_url TEXT, title TEXT NOT NULL, source_feed TEXT NOT NULL, published_at TEXT, rss_snippet TEXT, full_text TEXT, hn_comments TEXT, llm_score REAL, relevance_score REAL, relevance_reason TEXT, summary_json TEXT, topics_json TEXT, status TEXT NOT NULL DEFAULT 'new', created_at TEXT NOT NULL, delivered_at TEXT);
        CREATE TABLE topic_weights (topic TEXT PRIMARY KEY, category TEXT NOT NULL DEFAULT 'core', weight REAL NOT NULL DEFAULT 1.0, votes INTEGER NOT NULL DEFAULT 0, ignores INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
        CREATE TABLE polls (poll_id TEXT PRIMARY KEY, chat_id INTEGER NOT NULL, poll_type TEXT NOT NULL, options_metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, answered_at TEXT);
        CREATE TABLE state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);

        INSERT INTO state VALUES ('active_chat_id', '424242', '2026-11-01T00:00:00+00:00');
        INSERT INTO state VALUES ('preference_summary', 'Loves ML compilers.', '2026-11-01T00:00:00+00:00');
        INSERT INTO state VALUES ('digests_delivered_count', '7', '2026-11-01T00:00:00+00:00');
        INSERT INTO topic_weights VALUES ('AI Agents & Tool Use', 'core', 2.1, 4, 0, '2026-11-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    db.init_schema()

    migrated_user = db.get_user(424242)
    assert migrated_user is not None
    assert migrated_user["role"] == "admin"
    assert migrated_user["status"] == "active"
    assert migrated_user["preference_summary"] == "Loves ML compilers."
    assert migrated_user["digests_delivered_count"] == 7

    user_weights = {r["topic"]: r for r in db.get_user_topic_weights(424242)}
    assert user_weights["AI Agents & Tool Use"]["weight"] == 2.1


@pytest.mark.asyncio
async def test_digest_job_skips_per_user_quiet_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_org_settings(tmp_path)
    db = Database(tmp_path / "quiet_org.db")
    db.init_schema()
    db.seed_core_topics(CORE_TOPICS)
    db.upsert_user(1001, role="admin", status="active", core_topics=CORE_TOPICS)
    db.upsert_user(2001, role="member", status="active", core_topics=CORE_TOPICS)
    db.update_user_settings(1001, quiet_hours="00-23")  # in quiet hours
    db.update_user_settings(2001, quiet_hours=None)

    delivered_to: list[int] = []

    async def fake_deliver(bot: Any, chat_id: int, *args: Any, **kwargs: Any) -> dict[str, Any]:
        delivered_to.append(chat_id)
        return {"delivered": 1, "poll_sent": True, "discovery_sent": False}

    monkeypatch.setattr(bot_module, "deliver_digest", fake_deliver)
    monkeypatch.setattr(
        bot_module,
        "is_within_quiet_hours",
        lambda qh, now=None: qh == "00-23",
    )

    app = SimpleNamespace(
        bot_data={
            "settings": settings,
            "db": db,
            "llm": _CountingLLM(),
            "pipeline_lock": asyncio.Lock(),
            "profile_lock": asyncio.Lock(),
            "background_tasks": set(),
        }
    )
    await digest_job(SimpleNamespace(application=app, bot=_OrgFakeBot()))  # type: ignore[arg-type]
    assert delivered_to == [2001]
