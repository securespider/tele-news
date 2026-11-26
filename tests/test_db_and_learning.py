"""Tests for SQLite persistence, feed seeding, scoring adjustment, and poll learning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from db import Database
from learning import (
    CORE_TOPICS,
    compute_blended_score,
    process_poll_answer,
    resolve_canonical_topic,
    select_digest_articles,
)


def _init_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.init_schema()
    db.seed_core_topics(CORE_TOPICS)
    return db


def test_schema_seeding_and_soft_delete_persistence(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    feeds = ["https://hnrss.org/frontpage", "https://openai.com/news/rss.xml"]
    db.seed_feeds(feeds)
    assert len(db.get_enabled_feeds()) == 2

    # Soft-disable one feed, then simulate restart calling seed_feeds again
    assert db.remove_feed("https://openai.com/news/rss.xml") is True
    db.seed_feeds(feeds)

    enabled_urls = [f["url"] for f in db.get_enabled_feeds()]
    assert enabled_urls == ["https://hnrss.org/frontpage"]

    # Re-adding restores enabled=1
    assert db.add_feed("https://openai.com/news/rss.xml") is True
    assert len(db.get_enabled_feeds()) == 2


def test_url_dedup_and_comments_url_backfill(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    first_batch = [
        {
            "id": "art_1",
            "url": "https://openai.com/index/gpt5",
            "comments_url": None,
            "title": "OpenAI GPT-5 Launch",
            "source_feed": "https://openai.com/news/rss.xml",
            "published_at": "2026-11-22T10:00:00+00:00",
            "rss_snippet": "Official post",
        }
    ]
    assert db.insert_new_articles(first_batch) == 1

    # Duplicate from HN feed brings comments_url
    second_batch = [
        {
            "id": "art_1",
            "url": "https://openai.com/index/gpt5",
            "comments_url": "https://news.ycombinator.com/item?id=999111",
            "title": "OpenAI GPT-5 Launch",
            "source_feed": "https://hnrss.org/frontpage",
            "published_at": "2026-11-22T10:30:00+00:00",
            "rss_snippet": "HN link",
        }
    ]
    assert db.insert_new_articles(second_batch) == 0
    saved = db.get_article("art_1")
    assert saved is not None
    assert saved["comments_url"] == "https://news.ycombinator.com/item?id=999111"


def test_onboarding_poll_updates(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    meta = [{"topic": t} for t in CORE_TOPICS]
    db.save_poll("poll_onboard", 1001, "onboarding", meta)
    poll = db.get_poll("poll_onboard")
    assert poll is not None

    changed, boosted, update_profile = process_poll_answer(db, poll, [0, 1])
    assert changed is True
    assert update_profile is True
    assert set(boosted) == {CORE_TOPICS[0], CORE_TOPICS[1]}

    weights = {r["topic"]: r for r in db.get_all_topic_weights()}
    assert weights[CORE_TOPICS[0]]["weight"] >= 1.8
    assert weights[CORE_TOPICS[1]]["weight"] >= 1.8
    assert weights[CORE_TOPICS[2]]["weight"] <= 0.7


def test_digest_feedback_overlap_clamping_idempotency_and_no_unseen_junk(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    # Insert 2 delivered articles
    db.insert_new_articles(
        [
            {
                "id": "a1",
                "url": "https://example.com/1",
                "comments_url": None,
                "title": "MCP Agents Deep Dive",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": "2026-11-22T12:00:00+00:00",
                "rss_snippet": "Snippet 1",
            },
            {
                "id": "a2",
                "url": "https://example.com/2",
                "comments_url": None,
                "title": "GPU Cluster Policy",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": "2026-11-22T12:00:00+00:00",
                "rss_snippet": "Snippet 2",
            },
        ]
    )
    db.mark_article_delivered("a1", None, None, "{}", ["AI Agents & Tool Use", "MCP"])
    db.mark_article_delivered(
        "a2",
        None,
        None,
        "{}",
        ["AI Agents & Tool Use", "AI Hardware, Chips & Inference", "ObscureUnseenTag"],
    )

    meta = [
        {"article_id": "a1", "topics": ["AI Agents & Tool Use", "MCP"]},
        {
            "article_id": "a2",
            "topics": ["AI Agents & Tool Use", "AI Hardware, Chips & Inference", "ObscureUnseenTag"],
        },
        {"none_option": True},
    ]
    db.save_poll("poll_digest_1", 1001, "digest_feedback", meta)
    poll = db.get_poll("poll_digest_1")
    assert poll is not None

    # User votes for article 0 (a1)
    changed, boosted, update_profile = process_poll_answer(db, poll, [0])
    assert changed is True
    assert update_profile is True
    assert boosted == ["AI Agents & Tool Use", "MCP"]

    weights = {r["topic"]: r for r in db.get_all_topic_weights()}
    # Overlapping core topic gets +0.25 boost and NO -0.08 penalty
    assert weights["AI Agents & Tool Use"]["weight"] == 1.25
    assert weights["AI Agents & Tool Use"]["ignores"] == 0
    # New voted sub-topic is added as learned at 1.25
    assert weights["MCP"]["category"] == "learned"
    assert weights["MCP"]["weight"] == 1.25
    # Existing unvoted core topic gets -0.08 decay
    assert weights["AI Hardware, Chips & Inference"]["weight"] == 0.92
    assert weights["AI Hardware, Chips & Inference"]["ignores"] == 1
    # Unseen tag on unvoted article is NOT inserted as junk row
    assert "ObscureUnseenTag" not in weights

    # Articles user_feedback recorded
    assert db.get_article("a1")["user_feedback"] == 1
    assert db.get_article("a2")["user_feedback"] == -1
    liked, skipped = db.get_recent_titles_for_profile()
    assert liked == ["MCP Agents Deep Dive"]
    assert skipped == ["GPU Cluster Policy"]

    # Re-voting same poll_id is idempotent (ignored)
    changed_again, boosted_again, _ = process_poll_answer(db, poll, [1])
    assert changed_again is False
    assert boosted_again == []


def test_digest_feedback_none_of_these_decays_existing_and_clamps(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    db.upsert_topic_weight("AI Coding & DevTools", "core", 0.12)
    meta = [
        {"article_id": "a1", "topics": ["AI Coding & DevTools", "BrandNewSkippedTag"]},
        {"none_option": True},
    ]
    db.save_poll("poll_none", 1001, "digest_feedback", meta)
    poll = db.get_poll("poll_none")
    assert poll is not None

    changed, boosted, _ = process_poll_answer(db, poll, [1])
    assert changed is True
    assert boosted == []
    weights = {r["topic"]: r for r in db.get_all_topic_weights()}
    assert weights["AI Coding & DevTools"]["weight"] == 0.1  # clamped at MIN_WEIGHT
    assert "BrandNewSkippedTag" not in weights


def test_topic_discovery_and_mean_reversion(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    meta = [
        {"topic": "Test-Time Compute"},
        {"topic": "Browser-Use Agents"},
        {"none_option": True},
    ]
    db.save_poll("poll_disc", 1001, "topic_discovery", meta)
    poll = db.get_poll("poll_disc")
    assert poll is not None

    changed, boosted, update_profile = process_poll_answer(db, poll, [0])
    assert changed is True
    assert boosted == ["Test-Time Compute"]
    assert update_profile is True

    weights = {r["topic"]: r for r in db.get_all_topic_weights()}
    assert weights["Test-Time Compute"]["category"] == "learned"
    assert weights["Test-Time Compute"]["weight"] == 1.5
    assert "Browser-Use Agents" not in weights

    db.apply_weight_mean_reversion(0.98)
    weights_after = {r["topic"]: r for r in db.get_all_topic_weights()}
    assert weights_after["Test-Time Compute"]["weight"] == 1.49


def test_neutral_centered_compute_blended_score() -> None:
    weights = {
        "AI Agents & Tool Use": {"weight": 2.5},
        "AI Startups, Product & Policy": {"weight": 0.1},
        "LLMs & Foundation Models": {"weight": 1.0},
    }
    # Canonical case-insensitive resolution
    assert resolve_canonical_topic("  ai agents & tool use ", weights) == "AI Agents & Tool Use"

    # Cold start / neutral weight (1.0) leaves llm_score untouched
    assert compute_blended_score(0.65, ["LLMs & Foundation Models"], weights) == 0.65
    # Unseen tag only -> neutral
    assert compute_blended_score(0.72, ["Brand New Subtopic"], weights) == 0.72
    # Unseen sub-topic does NOT drag down known high-weight topic
    boosted_only = compute_blended_score(0.70, ["ai agents & tool use"], weights)
    boosted_with_unseen = compute_blended_score(
        0.70, ["AI Agents & Tool Use", "Brand New Subtopic"], weights
    )
    assert boosted_only == boosted_with_unseen
    assert boosted_only == 0.85  # 0.70 + 0.20 * ((2.5 - 1.0) / 2.0)

    # Low weight penalty & [0.0, 1.0] clamp
    penalized = compute_blended_score(0.70, ["AI Startups, Product & Policy"], weights)
    assert penalized == 0.5  # 0.70 - 0.20
    assert compute_blended_score(0.10, ["AI Startups, Product & Policy"], weights) == 0.0
    assert compute_blended_score(0.95, ["AI Agents & Tool Use"], weights) == 1.0


def test_expiration_recency_and_jaccard_dedup(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    now = datetime(2026, 11, 22, 18, 0, 0, tzinfo=timezone.utc)
    old_ts = (now - timedelta(hours=80)).isoformat()
    mid_ts = (now - timedelta(hours=10)).isoformat()
    fresh_ts = (now - timedelta(minutes=30)).isoformat()

    db.insert_new_articles(
        [
            {
                "id": "old_art",
                "url": "https://example.com/old",
                "comments_url": None,
                "title": "Old AI Story",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": old_ts,
                "rss_snippet": "Old",
            },
            {
                "id": "dup_1",
                "url": "https://example.com/d1",
                "comments_url": None,
                "title": "OpenAI Launches New Reasoning Model GPT-5",
                "source_feed": "https://openai.com/news/rss.xml",
                "published_at": fresh_ts,
                "rss_snippet": "Official",
            },
            {
                "id": "dup_2",
                "url": "https://techcrunch.com/d2",
                "comments_url": None,
                "title": "OpenAI Launches Reasoning Model GPT-5 Today",
                "source_feed": "https://techcrunch.com/feed",
                "published_at": mid_ts,
                "rss_snippet": "Coverage",
            },
            {
                "id": "distinct_3",
                "url": "https://example.com/d3",
                "comments_url": None,
                "title": "vLLM Adds FP8 Kernel Support for Blackwell GPUs",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": fresh_ts,
                "rss_snippet": "Inference",
            },
        ]
    )
    for art_id, score in [("old_art", 0.99), ("dup_1", 0.90), ("dup_2", 0.89), ("distinct_3", 0.80)]:
        db.update_article_stage1_score(art_id, score, score, "Relevant", ["AI"], "queued")

    expired_count = db.expire_old_queued_articles(max_age_hours=72, now=now)
    assert expired_count == 1
    assert db.get_article("old_art")["status"] == "expired"

    queued = db.get_queued_articles(limit=10)
    picked = select_digest_articles(queued, max_articles=2, max_age_hours=72, now=now)
    picked_ids = [item["id"] for item in picked]
    # dup_2 is skipped via Jaccard title similarity (> 0.55) with dup_1
    assert picked_ids == ["dup_1", "distinct_3"]


def test_reset_failed_articles_recovers_recent_and_expires_stale(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    now = datetime(2026, 11, 25, 9, 0, 0, tzinfo=timezone.utc)
    recent_ts = (now - timedelta(hours=12)).isoformat()
    stale_ts = (now - timedelta(hours=90)).isoformat()

    db.insert_new_articles(
        [
            {
                "id": "fail_recent",
                "url": "https://example.com/recent",
                "comments_url": None,
                "title": "Recent Failed AI Story",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": recent_ts,
                "rss_snippet": "Recent",
            },
            {
                "id": "fail_stale",
                "url": "https://example.com/stale",
                "comments_url": None,
                "title": "Stale Failed AI Story",
                "source_feed": "https://hnrss.org/frontpage",
                "published_at": stale_ts,
                "rss_snippet": "Stale",
            },
        ]
    )
    for _ in range(3):
        db.increment_stage1_attempts(["fail_recent", "fail_stale"], max_attempts=3)

    assert db.get_article("fail_recent")["status"] == "failed"
    assert db.get_article("fail_stale")["status"] == "failed"

    reset_count = db.reset_failed_articles(max_age_hours=72, now=now)
    assert reset_count == 1

    recent_row = db.get_article("fail_recent")
    stale_row = db.get_article("fail_stale")
    assert recent_row is not None and recent_row["status"] == "new"
    assert recent_row["stage1_attempts"] == 0
    assert stale_row is not None and stale_row["status"] == "expired"


def test_record_feed_error_auto_disable_and_reenable(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    feed_url = "https://dead.example/rss"
    db.seed_feeds([feed_url])

    # Threshold 0 never auto-disables
    count0, disabled0 = db.record_feed_error(feed_url, "timeout", max_consecutive_failures=0)
    assert (count0, disabled0) == (1, False)
    assert len(db.get_enabled_feeds()) == 1

    # Next two failures reach threshold=3 and soft-disable
    assert db.record_feed_error(feed_url, "timeout", max_consecutive_failures=3) == (2, False)
    assert db.record_feed_error(feed_url, "500 error", max_consecutive_failures=3) == (3, True)
    assert db.get_enabled_feeds() == []

    # Calling again while already disabled does not report a second transition
    assert db.record_feed_error(feed_url, "500 error", max_consecutive_failures=3) == (4, False)

    # Restart seed_feeds leaves auto-disabled feed disabled
    db.seed_feeds([feed_url])
    assert db.get_enabled_feeds() == []

    # Explicit add_feed re-enables and clears failure counters
    assert db.add_feed(feed_url, title="Recovered Feed") is True
    enabled = db.get_enabled_feeds()
    assert len(enabled) == 1
    assert enabled[0]["consecutive_failures"] == 0
    assert enabled[0]["last_error"] is None
    assert enabled[0]["title"] == "Recovered Feed"


def test_prune_old_articles_respects_status_window_and_batching(tmp_path: Path) -> None:
    db = _init_db(tmp_path)
    now = datetime(2026, 11, 26, 0, 0, 0, tzinfo=timezone.utc)
    old_ts = (now - timedelta(days=40)).isoformat()
    mid_ts = (now - timedelta(days=2)).isoformat()
    recent_ts = (now - timedelta(hours=6)).isoformat()

    rows = [
        ("old_expired", old_ts, "expired"),
        ("old_filtered", old_ts, "filtered_out"),
        ("old_failed", old_ts, "failed"),
        ("old_delivered", old_ts, "delivered"),
        ("old_queued", old_ts, "queued"),
        ("old_new", old_ts, "new"),
        ("mid_filtered", mid_ts, "filtered_out"),
        ("recent_expired", recent_ts, "expired"),
    ]
    with db._connect() as conn:
        for art_id, ts, status in rows:
            conn.execute(
                """
                INSERT INTO articles (id, url, title, source_feed, published_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (art_id, f"https://example.com/{art_id}", art_id, "https://hnrss.org/frontpage", ts, status, ts),
            )
        # Add >500 extra old expired rows to exercise 500-ID batching
        for i in range(510):
            art_id = f"bulk_old_{i}"
            conn.execute(
                """
                INSERT INTO articles (id, url, title, source_feed, published_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'expired', ?);
                """,
                (art_id, f"https://example.com/{art_id}", art_id, "https://hnrss.org/frontpage", old_ts, old_ts),
            )

    # retention_days=0 disables pruning
    assert db.prune_old_articles(0, now=now) == 0

    # min_age_hours=72 protects mid_filtered (2 days old) even if retention_days=1
    pruned = db.prune_old_articles(30, min_age_hours=72, now=now)
    assert pruned == 3 + 510

    assert db.get_article("old_expired") is None
    assert db.get_article("old_filtered") is None
    assert db.get_article("old_failed") is None
    assert db.get_article("old_delivered") is not None
    assert db.get_article("old_queued") is not None
    assert db.get_article("old_new") is not None
    assert db.get_article("mid_filtered") is not None
    assert db.get_article("recent_expired") is not None

    # Even with retention_days=1, min_age_hours=72 protects 2-day-old mid_filtered
    assert db.prune_old_articles(1, min_age_hours=72, now=now) == 0
    assert db.get_article("mid_filtered") is not None
