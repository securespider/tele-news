"""Telegram bot command handlers, RBAC, signed callbacks, literature review & scheduler."""

from __future__ import annotations

import asyncio
import html
import io
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from telegram import Bot, BotCommand, InputFile, LinkPreviewOptions, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    PollAnswerHandler,
)

from config import QUIET_HOURS_RE, Settings
from db import Database, parse_iso_utc, utc_now_iso
from formatting import (
    MAX_ARTICLE_HTML_CHARS,
    MAX_TELEGRAM_CHUNK_CHARS,
    _join_html_lines,
    _safe_href,
    build_article_action_keyboard,
    build_digest_quick_keyboard,
    chunk_digest_messages,
    format_article_html,
    format_library_html,
    format_litreview_html,
    format_paper_comparison_html,
    format_paper_review_html,
    format_trending_html,
    format_weight_bar,
)
from learning import (
    CORE_TOPICS,
    DEFAULT_PREFERENCE_SUMMARY,
    build_profile_context,
    build_safe_poll_options,
    build_user_profile_context,
    compute_blended_score,
    process_poll_answer,
    rank_candidates_for_user,
    resolve_canonical_topic,
    select_digest_articles,
    truncate_question,
)
from llm import BudgetExceededError, LLMClient, PaperReview, Stage2Summary
from papers import (
    extract_arxiv_id,
    fetch_hf_daily_papers,
    format_bibtex,
    format_library_markdown,
    format_litreview_markdown,
    paper_record_to_article_dict,
    resolve_paper,
    search_arxiv,
    search_semantic_scholar,
)
from scraper import (
    compute_article_id,
    enrich_articles_batch,
    fetch_full_text,
    fetch_new_rss_entries,
    is_safe_public_url,
    normalize_url,
    validate_rss_feed_url,
)
from security import (
    CallbackSigner,
    SlidingWindowRateLimiter,
    TelegramRateLimiter,
    generate_invite_token,
    hash_invite_token,
)

logger = logging.getLogger(__name__)

_ROLE_LEVEL = {"member": 1, "admin": 2}
_VALID_DIGEST_MODES = {"balanced", "papers", "news"}


def is_within_quiet_hours(quiet_hours: str | None, now: datetime | None = None) -> bool:
    """Return True if local time hour falls within the configured HH-HH quiet window."""
    if not quiet_hours:
        return False
    parts = quiet_hours.split("-")
    if len(parts) != 2:
        return False
    start_h, end_h = int(parts[0]), int(parts[1])
    hour = (now or datetime.now()).hour
    if start_h < end_h:
        return start_h <= hour < end_h
    return hour >= start_h or hour < end_h


def _extract_user_meta(update: Update) -> tuple[int | None, str | None, str | None]:
    user = getattr(update, "effective_user", None)
    if user is not None:
        uid = getattr(user, "id", None)
        uname = getattr(user, "username", None)
        first = getattr(user, "first_name", None) or ""
        last = getattr(user, "last_name", None) or ""
        full = " ".join(p for p in (first, last) if p).strip() or uname
        return (uid, uname, full)
    chat = getattr(update, "effective_chat", None)
    if chat is not None:
        return (getattr(chat, "id", None), None, None)
    return (None, None, None)


def _command_head(update: Update) -> str:
    msg = getattr(update, "effective_message", None)
    text = (getattr(msg, "text", None) or "").strip()
    if not text:
        return ""
    return text.split()[0].split("@")[0].lower()


def authorize_user(
    update: Update,
    settings: Settings,
    db: Database,
    *,
    min_role: str = "member",
) -> dict[str, Any] | None:
    """Authenticate and authorize private DM users or PollAnswer actors with RBAC."""
    # Enforce explicit legacy lock if configured alongside no extra org users
    explicit_single_lock = (
        settings.telegram_allowed_chat_id is not None
        and not settings.telegram_admin_ids
        and not settings.telegram_allowed_user_ids
    )

    if getattr(update, "poll_answer", None) is not None:
        pa_user = update.poll_answer.user
        if pa_user is None:
            return None
        uid = pa_user.id
        if explicit_single_lock and uid != settings.telegram_allowed_chat_id:
            return None
        user_row = db.get_user(uid)
        if user_row and user_row["status"] == "active":
            if _ROLE_LEVEL.get(user_row["role"], 0) >= _ROLE_LEVEL.get(min_role, 1):
                return user_row
            return None
        if settings.telegram_allowed_chat_id is not None and uid == settings.telegram_allowed_chat_id:
            return db.upsert_user(uid, role="admin", status="active", core_topics=CORE_TOPICS)
        active = db.get_state("active_chat_id")
        if active is not None and str(uid) == active:
            return db.upsert_user(uid, role="admin", status="active", core_topics=CORE_TOPICS)
        return None

    chat = getattr(update, "effective_chat", None)
    if chat is None or chat.type != ChatType.PRIVATE:
        return None

    uid, uname, display_name = _extract_user_meta(update)
    if uid is None:
        return None

    if explicit_single_lock and uid != settings.telegram_allowed_chat_id:
        return None

    user_row = db.get_user(uid)
    if user_row is not None:
        if user_row["status"] != "active":
            return None
        if _ROLE_LEVEL.get(user_row["role"], 0) < _ROLE_LEVEL.get(min_role, 1):
            return None
        db.touch_user_active(uid, uname, display_name)
        return user_row

    # Check static env allowlists (in case DB wasn't pre-seeded)
    if uid in settings.telegram_admin_ids or (
        settings.telegram_allowed_chat_id is not None and uid == settings.telegram_allowed_chat_id
    ):
        created = db.upsert_user(
            uid, username=uname, display_name=display_name, role="admin", status="active", core_topics=CORE_TOPICS
        )
        return created if _ROLE_LEVEL.get(created["role"], 0) >= _ROLE_LEVEL.get(min_role, 1) else None

    if uid in settings.telegram_allowed_user_ids:
        created = db.upsert_user(
            uid, username=uname, display_name=display_name, role="member", status="active", core_topics=CORE_TOPICS
        )
        return created if _ROLE_LEVEL.get(created["role"], 0) >= _ROLE_LEVEL.get(min_role, 1) else None

    # Legacy single-user auto-bind on /start when unconfigured and no active users exist
    cmd = _command_head(update)
    has_static_allowlist = bool(
        settings.telegram_allowed_chat_id is not None
        or settings.telegram_admin_ids
        or settings.telegram_allowed_user_ids
    )
    if (
        cmd == "/start"
        and settings.allow_self_register_first_admin
        and not has_static_allowlist
        and db.count_active_users() == 0
        and db.get_state("active_chat_id") is None
    ):
        db.set_state("active_chat_id", str(chat.id))
        created = db.upsert_user(
            uid, username=uname, display_name=display_name, role="admin", status="active", core_topics=CORE_TOPICS
        )
        db.log_audit_event(uid, "bootstrap_first_admin", str(uid))
        return created

    return None


def is_authorized(update: Update, settings: Settings, db: Database) -> bool:
    """Backward-compatible authorization helper wrapping authorize_user."""
    return authorize_user(update, settings, db, min_role="member") is not None


def _get_deps(
    context: ContextTypes.DEFAULT_TYPE,
) -> tuple[
    Settings,
    Database,
    LLMClient,
    asyncio.Lock,
    asyncio.Lock,
    set[asyncio.Task[Any]],
]:
    data = context.application.bot_data
    settings: Settings = data["settings"]
    if "command_limiter" not in data:
        data["command_limiter"] = SlidingWindowRateLimiter()
    if "llm_limiter" not in data:
        data["llm_limiter"] = SlidingWindowRateLimiter()
    if "callback_signer" not in data:
        data["callback_signer"] = CallbackSigner(settings.telegram_bot_token)
    return (
        settings,
        data["db"],
        data["llm"],
        data["pipeline_lock"],
        data["profile_lock"],
        data["background_tasks"],
    )


def _get_security_helpers(
    context: ContextTypes.DEFAULT_TYPE, settings: Settings
) -> tuple[SlidingWindowRateLimiter, SlidingWindowRateLimiter, CallbackSigner]:
    data = context.application.bot_data
    if "command_limiter" not in data:
        data["command_limiter"] = SlidingWindowRateLimiter()
    if "llm_limiter" not in data:
        data["llm_limiter"] = SlidingWindowRateLimiter()
    if "callback_signer" not in data:
        data["callback_signer"] = CallbackSigner(settings.telegram_bot_token)
    return data["command_limiter"], data["llm_limiter"], data["callback_signer"]


def _track_background_task(
    tasks: set[asyncio.Task[Any]], coro: Any
) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    tasks.add(task)

    def _on_done(done_task: asyncio.Task[Any]) -> None:
        tasks.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc is not None:
            logger.error("Background task failed: %s", exc, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


async def _require_auth_and_limits(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    min_role: str = "member",
    heavy_llm: bool = False,
) -> dict[str, Any] | None:
    settings, db, _, _, _, _ = _get_deps(context)
    cmd_limiter, llm_limiter, _ = _get_security_helpers(context, settings)
    uid, _, _ = _extract_user_meta(update)
    chat = getattr(update, "effective_chat", None)
    msg = getattr(update, "effective_message", None)

    if chat is None or chat.type != ChatType.PRIVATE:
        return None

    user = authorize_user(update, settings, db, min_role=min_role)
    if user is None:
        if uid is not None:
            existing = db.get_user(uid)
            if existing and existing["status"] == "active" and min_role == "admin":
                if msg is not None:
                    await msg.reply_text("🚫 Admin privileges are required for this command.")
                db.log_audit_event(uid, "rbac_denied", _command_head(update))
                return None

            allowed_warn, _ = cmd_limiter.check(f"unauth:{uid}", 3, 300.0)
            if allowed_warn:
                db.log_audit_event(uid, "unauthorized_access", _command_head(update))
                if msg is not None:
                    await msg.reply_text(
                        f"🔒 Access to <b>{html.escape(settings.org_name)}</b> requires an invitation.\n"
                        "Ask an org admin for an invite code and run <code>/join &lt;code&gt;</code>.",
                        parse_mode=ParseMode.HTML,
                    )
        return None

    ok_cmd, retry_cmd = cmd_limiter.check(
        f"cmd:{user['user_id']}", settings.rate_limit_commands_per_min, 60.0
    )
    if not ok_cmd:
        if msg is not None:
            await msg.reply_text(f"⏳ Rate limit reached. Please wait {retry_cmd:.0f}s.")
        return None

    if heavy_llm:
        ok_llm, retry_llm = llm_limiter.check(
            f"llm:{user['user_id']}", settings.rate_limit_llm_per_user_hour, 3600.0
        )
        if not ok_llm:
            if msg is not None:
                await msg.reply_text(
                    f"⏳ Hourly AI research quota reached. Try again in {int(retry_llm // 60) + 1} min."
                )
            return None
        ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
        if not ok_budget:
            if msg is not None:
                await msg.reply_text(
                    "⚠️ Organization daily LLM budget reached. Cached reviews & library remain available."
                )
            return None

    return user


async def _refresh_profile_summary(
    db: Database,
    llm: LLMClient,
    profile_lock: asyncio.Lock,
    *,
    user_id: int | None = None,
) -> None:
    async with profile_lock:
        if user_id is not None:
            current_summary = db.get_user_preference_summary(user_id) or DEFAULT_PREFERENCE_SUMMARY
            rows = db.get_user_topic_weights(user_id)
            liked_titles, skipped_titles = db.get_user_recent_titles_for_profile(user_id, limit=10)
        else:
            current_summary = db.get_state("preference_summary") or DEFAULT_PREFERENCE_SUMMARY
            rows = db.get_all_topic_weights()
            liked_titles, skipped_titles = db.get_recent_titles_for_profile(limit=10)

        top_topics = [(r["topic"], float(r["weight"])) for r in rows[:8]]
        low_topics = [
            (r["topic"], float(r["weight"])) for r in reversed(rows) if float(r["weight"]) < 0.9
        ][:5]
        new_summary = await llm.update_preference_profile(
            current_summary=current_summary,
            top_topics=top_topics,
            low_topics=low_topics,
            liked_titles=liked_titles,
            ignored_titles=skipped_titles,
        )
        if new_summary:
            if user_id is not None:
                db.set_user_preference_summary(user_id, new_summary)
            else:
                db.set_state("preference_summary", new_summary)


async def notify_auto_disabled_feeds(
    bot: Bot, chat_id: int, disabled_feeds: list[dict[str, Any]]
) -> None:
    """Send a Telegram alert when feeds are auto-disabled after repeated failures."""
    if not disabled_feeds:
        return
    lines = ["<b>⚠️ Auto-Disabled Failing RSS Feeds</b>"]
    for item in disabled_feeds:
        url = html.escape(str(item.get("url") or ""))
        failures = int(item.get("failures") or 0)
        err = html.escape(str(item.get("error") or "Unknown error")[:160])
        lines.append(f"• <code>{url}</code> ({failures} consecutive failures): {err}")
    lines.append("\nRe-enable once fixed with <code>/feeds add &lt;url&gt;</code>.")
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=_join_html_lines(lines),
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
    except Exception as exc:
        logger.warning("Failed sending auto-disable feed alert: %s", exc)


async def run_scrape_cycle(
    db: Database,
    llm: LLMClient,
    settings: Settings,
    *,
    include_hf_papers: bool = False,
) -> dict[str, Any]:
    """Fetch enabled feeds (and optionally HF Daily Papers), insert unseen items, and run Stage 1."""
    feeds = db.get_enabled_feeds()
    entries, feed_updates = await fetch_new_rss_entries(
        feeds,
        max_age_hours=settings.max_article_age_hours,
        max_entries_per_feed=settings.max_entries_per_feed,
    )

    if include_hf_papers and settings.enable_hf_daily_papers:
        hf_entries = await fetch_hf_daily_papers(limit=25)
        entries.extend(hf_entries)

    auto_disabled_feeds: list[dict[str, Any]] = []
    for feed_url, status_info in feed_updates.items():
        if status_info.get("ok"):
            db.update_feed_checked(
                feed_url,
                etag=status_info.get("etag"),
                last_modified=status_info.get("last_modified"),
                title=status_info.get("title"),
            )
        else:
            err_text = str(status_info.get("error") or "Unknown fetch error")
            failures, auto_disabled = db.record_feed_error(
                feed_url,
                err_text,
                max_consecutive_failures=settings.feed_max_consecutive_failures,
            )
            if auto_disabled:
                auto_disabled_feeds.append(
                    {"url": feed_url, "failures": failures, "error": err_text}
                )

    inserted = db.insert_new_articles(entries)
    new_articles = db.get_new_articles(limit=60)

    scored = 0
    queued = 0
    errors = 0

    if new_articles:
        profile = build_profile_context(db)
        rows = db.get_all_topic_weights()
        weights_map = {row["topic"]: dict(row) for row in rows}
        sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _score_one(article: dict[str, Any]) -> tuple[bool, bool]:
            ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
            if not ok_budget:
                return (False, False)
            try:
                async with sem:
                    if hasattr(llm, "score_candidate"):
                        verdict = await llm.score_candidate(article, profile)
                    else:
                        verdicts = await llm.score_candidates([article], profile)
                        verdict = verdicts[article["id"]]
            except Exception as exc:
                logger.warning(
                    "Stage 1 LLM scoring failed for %s (%s): %s",
                    article["id"],
                    article.get("title", "")[:80],
                    exc,
                )
                db.increment_stage1_attempts([article["id"]], max_attempts=3)
                return (False, False)

            canonical_topics: list[str] = []
            for raw_t in verdict.topics:
                canon = resolve_canonical_topic(raw_t, weights_map)
                if canon and canon not in canonical_topics:
                    canonical_topics.append(canon)

            blended = compute_blended_score(verdict.llm_score, canonical_topics, weights_map)
            # Keep high-upvote HF papers in queued pool even if user global weights are neutral
            hf_up = int(article.get("hf_upvotes") or 0)
            is_queued = verdict.is_ai and (
                blended >= settings.min_relevance_score
                or (hf_up >= 15 and verdict.llm_score >= 0.5)
            )
            status = "queued" if is_queued else "filtered_out"
            db.update_article_stage1_score(
                article_id=article["id"],
                llm_score=verdict.llm_score,
                relevance_score=blended,
                reason=verdict.reason,
                topics=canonical_topics,
                status=status,
            )
            return (True, status == "queued")

        outcomes = await asyncio.gather(*(_score_one(art) for art in new_articles))
        for ok, was_queued in outcomes:
            if ok:
                scored += 1
                if was_queued:
                    queued += 1
            else:
                errors += 1

    db.set_state("last_scrape_at", utc_now_iso())
    return {
        "feeds_checked": len(feeds),
        "inserted": inserted,
        "scored": scored,
        "queued": queued,
        "errors": errors,
        "auto_disabled": len(auto_disabled_feeds),
        "auto_disabled_feeds": auto_disabled_feeds,
    }


async def run_hf_papers_cycle(
    db: Database, llm: LLMClient, settings: Settings
) -> dict[str, Any]:
    """Ingest Hugging Face Daily Papers and run Stage 1 scoring on unseen papers."""
    if not settings.enable_hf_daily_papers:
        return {"fetched": 0, "inserted": 0, "queued": 0}
    hf_entries = await fetch_hf_daily_papers(limit=25)
    inserted = db.insert_new_articles(hf_entries)
    new_articles = [a for a in db.get_new_articles(limit=30) if a.get("content_type") == "paper"]

    queued = 0
    if new_articles:
        profile = build_profile_context(db)
        rows = db.get_all_topic_weights()
        weights_map = {row["topic"]: dict(row) for row in rows}
        sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

        async def _score_paper(article: dict[str, Any]) -> bool:
            ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
            if not ok_budget:
                return False
            try:
                async with sem:
                    if hasattr(llm, "score_candidate"):
                        verdict = await llm.score_candidate(article, profile)
                    else:
                        verdicts = await llm.score_candidates([article], profile)
                        verdict = verdicts[article["id"]]
            except Exception as exc:
                logger.warning("HF paper Stage 1 failed for %s: %s", article["id"], exc)
                db.increment_stage1_attempts([article["id"]], max_attempts=3)
                return False

            canonical_topics = [
                resolve_canonical_topic(t, weights_map) for t in verdict.topics if t
            ]
            blended = compute_blended_score(verdict.llm_score, canonical_topics, weights_map)
            status = "queued" if (verdict.is_ai and blended >= 0.45) else "filtered_out"
            db.update_article_stage1_score(
                article_id=article["id"],
                llm_score=verdict.llm_score,
                relevance_score=blended,
                reason=verdict.reason,
                topics=canonical_topics,
                status=status,
            )
            return status == "queued"

        results = await asyncio.gather(*(_score_paper(a) for a in new_articles))
        queued = sum(1 for r in results if r)

    db.set_state("last_hf_scrape_at", utc_now_iso())
    return {"fetched": len(hf_entries), "inserted": inserted, "queued": queued}


async def send_onboarding_poll(
    bot: Bot, chat_id: int, db: Database, *, user_id: int | None = None
) -> None:
    """Send the 8-topic non-anonymous multi-select onboarding poll and persist it."""
    question = truncate_question(
        "Which core AI topics do you want prioritized? (Select all that interest you)"
    )
    options = list(CORE_TOPICS)
    metadata = [{"topic": topic} for topic in CORE_TOPICS]
    poll_msg = await bot.send_poll(
        chat_id=chat_id,
        question=question,
        options=options,
        is_anonymous=False,
        allows_multiple_answers=True,
    )
    db.save_poll(
        poll_id=poll_msg.poll.id,
        chat_id=chat_id,
        poll_type="onboarding",
        options_metadata=metadata,
        message_id=poll_msg.message_id,
        user_id=user_id,
    )


async def send_discovery_poll(
    bot: Bot,
    chat_id: int,
    db: Database,
    llm: LLMClient,
    *,
    user_id: int | None = None,
) -> bool:
    """Propose emerging AI sub-topics via LLM and send a discovery poll if non-empty."""
    recent_titles = db.get_recent_ai_titles(limit=30)
    if not recent_titles:
        return False

    if user_id is not None:
        existing_topics = [row["topic"] for row in db.get_user_topic_weights(user_id)]
    else:
        existing_topics = [row["topic"] for row in db.get_all_topic_weights()]

    proposed = await llm.propose_discovery_topics(recent_titles, existing_topics)
    items = [(topic, {"topic": topic}) for topic in proposed]
    options, metadata = build_safe_poll_options(items, sentinel_label="None of these")
    if not options:
        return False

    question = truncate_question(
        "Emerging AI sub-topics spotted recently — tap any you'd like to see more of:"
    )
    poll_msg = await bot.send_poll(
        chat_id=chat_id,
        question=question,
        options=options,
        is_anonymous=False,
        allows_multiple_answers=True,
    )
    db.save_poll(
        poll_id=poll_msg.poll.id,
        chat_id=chat_id,
        poll_type="topic_discovery",
        options_metadata=metadata,
        message_id=poll_msg.message_id,
        user_id=user_id,
    )
    return True


def _decode_cached_summary(article: dict[str, Any]) -> Stage2Summary | None:
    raw = article.get("summary_json")
    if not raw:
        return None
    try:
        return Stage2Summary.model_validate_json(str(raw))
    except Exception:
        return None


async def deliver_digest(
    bot: Bot,
    chat_id: int,
    db: Database,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: int | None = None,
    mode_override: str | None = None,
    signer: CallbackSigner | None = None,
) -> dict[str, Any]:
    """Expire stale queued items, personalize Top-K, share cached summaries, send digest + poll."""
    db.expire_old_queued_articles(settings.max_article_age_hours)

    if user_id is not None:
        user_row = db.get_user(user_id)
        limit = int(
            (user_row or {}).get("max_articles_per_digest") or settings.max_articles_per_digest
        )
        mode = mode_override or (user_row or {}).get("digest_mode") or settings.default_digest_mode
        candidates = db.get_candidate_articles_for_user(
            user_id,
            max_age_hours=settings.max_article_age_hours,
            content_mode=mode,
            limit=max(20, limit * 5),
        )
        user_weights = {r["topic"]: dict(r) for r in db.get_user_topic_weights(user_id)}
        selected = rank_candidates_for_user(
            candidates,
            user_weights,
            limit,
            min_relevance_score=settings.min_relevance_score,
            max_age_hours=settings.max_article_age_hours,
        )
        profile = build_user_profile_context(db, user_id)
        weights_map = user_weights
    else:
        candidates = db.get_queued_articles(limit=settings.max_articles_per_digest * 3)
        selected = select_digest_articles(
            candidates,
            settings.max_articles_per_digest,
            max_age_hours=settings.max_article_age_hours,
        )
        profile = build_profile_context(db)
        weights_map = {row["topic"]: dict(row) for row in db.get_all_topic_weights()}

    if not selected:
        return {"delivered": 0, "poll_sent": False, "discovery_sent": False}

    uncached = [art for art in selected if _decode_cached_summary(art) is None]
    enriched = (
        await enrich_articles_batch(uncached, max_concurrency=3) if uncached else {}
    )

    sem = asyncio.Semaphore(max(1, settings.llm_max_concurrency))

    async def _prepare_one(
        article: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, str | None, Stage2Summary, list[str], bool]:
        cached = _decode_cached_summary(article)
        if cached is not None:
            canon_topics = cached.topics or ["AI"]
            return (
                article,
                article.get("full_text"),
                article.get("hn_comments"),
                cached,
                canon_topics,
                True,
            )

        full_text, hn_comments = enriched.get(article["id"], (None, None))
        enriched_article = {**article, "full_text": full_text, "hn_comments": hn_comments}
        db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
        async with sem:
            summary = await llm.summarize_article(enriched_article, profile)
        canon_topics: list[str] = []
        for raw_t in summary.topics:
            canon = resolve_canonical_topic(raw_t, weights_map)
            if canon and canon not in canon_topics:
                canon_topics.append(canon)
        summary.topics = canon_topics or ["AI"]
        return (article, full_text, hn_comments, summary, summary.topics, False)

    prepared = await asyncio.gather(*(_prepare_one(art) for art in selected))

    blocks = [
        format_article_html(idx, article, summary)
        for idx, (article, _, _, summary, _, _) in enumerate(prepared, start=1)
    ]
    header = f"<b>🤖 DailyAI Digest ({len(prepared)} {'story' if len(prepared) == 1 else 'stories'})</b>"
    chunks = chunk_digest_messages(header, blocks, max_len=MAX_TELEGRAM_CHUNK_CHARS)

    reply_markup = (
        build_digest_quick_keyboard(signer, user_id, [p[0] for p in prepared])
        if (signer is not None and user_id is not None)
        else None
    )

    for idx, chunk in enumerate(chunks):
        send_kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": ParseMode.HTML,
            "link_preview_options": LinkPreviewOptions(is_disabled=True),
        }
        if reply_markup is not None and idx == len(chunks) - 1:
            send_kwargs["reply_markup"] = reply_markup
        await bot.send_message(**send_kwargs)

    poll_items: list[tuple[str, dict[str, Any]]] = []
    delivered_ids: list[str] = []
    for article, full_text, hn_comments, summary, topics, was_cached in prepared:
        delivered_ids.append(article["id"])
        if user_id is not None:
            if not was_cached:
                db.cache_article_summary(
                    article_id=article["id"],
                    full_text=full_text,
                    hn_comments=hn_comments,
                    summary_json=summary.model_dump_json(),
                    topics=topics,
                )
        else:
            db.mark_article_delivered(
                article_id=article["id"],
                full_text=full_text,
                hn_comments=hn_comments,
                summary_json=summary.model_dump_json(),
                topics=topics,
            )
        poll_items.append(
            (
                summary.poll_label or article["title"],
                {"article_id": article["id"], "topics": topics},
            )
        )

    if user_id is not None:
        db.record_user_deliveries(user_id, delivered_ids)

    options, metadata = build_safe_poll_options(
        poll_items, sentinel_label="None of these / More variety"
    )
    poll_sent = False
    if options:
        question = truncate_question("Which stories from this digest were most valuable to you?")
        try:
            poll_msg = await bot.send_poll(
                chat_id=chat_id,
                question=question,
                options=options,
                is_anonymous=False,
                allows_multiple_answers=True,
            )
            db.save_poll(
                poll_id=poll_msg.poll.id,
                chat_id=chat_id,
                poll_type="digest_feedback",
                options_metadata=metadata,
                message_id=poll_msg.message_id,
                user_id=user_id,
            )
            poll_sent = True
        except Exception as exc:
            logger.warning("Failed sending post-digest feedback poll: %s", exc)

    if user_id is not None:
        new_count = db.increment_user_digest_count(user_id)
        if new_count % 5 == 0:
            db.apply_user_weight_mean_reversion(user_id, 0.98)
    else:
        prev_count = int(db.get_state("digests_delivered_count", "0") or 0)
        new_count = prev_count + 1
        db.set_state("digests_delivered_count", str(new_count))
        if new_count % 5 == 0:
            db.apply_weight_mean_reversion(0.98)

    discovery_sent = False
    if new_count % settings.discovery_poll_every_n_digests == 0:
        try:
            discovery_sent = await send_discovery_poll(
                bot, chat_id, db, llm, user_id=user_id
            )
        except Exception as exc:
            logger.warning("Scheduled discovery poll failed: %s", exc)

    return {
        "delivered": len(prepared),
        "poll_sent": poll_sent,
        "discovery_sent": discovery_sent,
    }


async def _resolve_paper_ref(
    ref: str,
    *,
    user_id: int,
    db: Database,
    settings: Settings,
) -> dict[str, Any] | None:
    """Resolve an article_id, library #id, arXiv ID, or URL to a DB article dict."""
    cleaned = (ref or "").strip()
    if not cleaned:
        return None

    if cleaned.isdigit():
        lib_item = db.get_library_item(user_id, int(cleaned))
        if lib_item is not None:
            art = db.get_article(lib_item["article_id"])
            if art is not None:
                return art

    direct = db.get_article(cleaned)
    if direct is not None:
        return direct

    by_prefix = db.get_article_by_prefix(cleaned)
    if by_prefix is not None:
        return by_prefix

    arxiv_id = extract_arxiv_id(cleaned)
    if arxiv_id:
        by_arxiv = db.get_article_by_arxiv_id(arxiv_id)
        if by_arxiv is not None:
            return by_arxiv

    if is_safe_public_url(cleaned):
        by_url = db.get_article(compute_article_id(cleaned))
        if by_url is not None:
            return by_url

    if arxiv_id or len(cleaned) >= 4:
        paper = await resolve_paper(
            cleaned,
            api_key=settings.semantic_scholar_api_key,
            throttle_seconds=3.0 if settings.enable_arxiv_api else 0.0,
        )
        if paper is not None:
            art_dict = paper_record_to_article_dict(paper)
            art_id = db.upsert_paper_article(art_dict)
            return db.get_article(art_id)

    return None


# ------------------------------------------------------------------
# Command Handlers
# ------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])
    db.seed_user_core_topics(uid, CORE_TOPICS)

    role_badge = "🛡️ Admin" if user["role"] == "admin" else "👤 Member"
    welcome = (
        f"<b>👋 Welcome to {html.escape(settings.org_name)} AI Agent!</b> ({role_badge})\n\n"
        "I curate personalized AI news &amp; academic paper digests, run structured peer reviews, "
        "synthesize grounded literature reviews, and maintain your personal BibTeX research library.\n\n"
        "<b>Daily Digest &amp; Learning:</b>\n"
        "• /digest [balanced|papers|news] — Personalized AI briefing\n"
        "• /topics — Your learned interest profile &amp; weights\n"
        "• /poll / /discover — Tune core or emerging AI topics\n"
        "• /settings — Cadence, digest size, mode &amp; quiet hours\n\n"
        "<b>Literature Review &amp; Library:</b>\n"
        "• /review &lt;arxiv_id|url|lib_id&gt; — Structured paper critique\n"
        "• /litreview &lt;question&gt; — Multi-paper synthesis + .md/.bib\n"
        "• /compare &lt;id1&gt; &lt;id2&gt; — Side-by-side paper matrix\n"
        "• /search &lt;query&gt; — Search org corpus &amp; arXiv\n"
        "• /trending — HF Daily Papers &amp; org-wide top picks\n"
        "• /library, /save, /note, /read, /bibtex, /export\n"
        "• /help — Full command reference"
    )
    await update.effective_message.reply_text(welcome, parse_mode=ParseMode.HTML)
    await send_onboarding_poll(context.bot, update.effective_chat.id, db, user_id=uid)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return

    lines = [
        "<b>📖 Command Reference</b>",
        "",
        "<b>Digest &amp; Preferences:</b>",
        "• <code>/digest [balanced|papers|news]</code> — Get your personalized digest",
        "• <code>/topics</code> — View topic weights &amp; profile summary",
        "• <code>/poll</code> — Re-run 8-topic onboarding poll",
        "• <code>/discover</code> — Vote on emerging AI sub-topics",
        "• <code>/settings [cadence H|max N|quiet HH-HH|off|mode M]</code>",
        "• <code>/pause</code> / <code>/resume</code> — Pause or resume scheduled digests",
        "",
        "<b>Literature Review &amp; Papers:</b>",
        "• <code>/review &lt;arxiv_id|url|lib_id&gt;</code> — Methodology, results &amp; limitations",
        "• <code>/litreview &lt;research question&gt;</code> — Grounded synthesis + .md &amp; .bib export",
        "• <code>/compare &lt;id1&gt; &lt;id2&gt; [id3] [id4]</code> — Compare papers",
        "• <code>/search [--arxiv] &lt;query&gt;</code> — Search corpus &amp; arXiv",
        "• <code>/trending</code> — HF Daily Papers &amp; team favorites",
        "",
        "<b>Reading Library:</b>",
        "• <code>/library [to_read|reading|read|tag]</code>",
        "• <code>/save &lt;arxiv_id|url|id&gt; [tag1,tag2]</code>",
        "• <code>/note &lt;lib_id&gt; &lt;note&gt;</code> | <code>/read &lt;lib_id&gt;</code>",
        "• <code>/bibtex &lt;arxiv_id|lib_id&gt;</code> | <code>/export [bibtex|markdown]</code>",
        "",
        "<b>Operations:</b>",
        "• <code>/feeds</code> — View RSS feeds (admins: add/remove)",
        "• <code>/status</code> — System, corpus &amp; budget status",
    ]
    if user["role"] == "admin":
        lines.extend(
            [
                "",
                "<b>Admin RBAC:</b>",
                "• <code>/admin invite [member|admin] [max_uses] [hours]</code>",
                "• <code>/admin users | approve &lt;id&gt; | promote &lt;id&gt; | demote &lt;id&gt; | revoke &lt;id&gt;</code>",
                "• <code>/admin audit | budget</code>",
            ]
        )
    await update.effective_message.reply_text(_join_html_lines(lines), parse_mode=ParseMode.HTML)


async def cmd_join(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, _, _, _, _ = _get_deps(context)
    cmd_limiter, _, _ = _get_security_helpers(context, settings)
    chat = getattr(update, "effective_chat", None)
    msg = getattr(update, "effective_message", None)
    if chat is None or chat.type != ChatType.PRIVATE or msg is None:
        return

    uid, uname, display_name = _extract_user_meta(update)
    if uid is None:
        return

    ok_try, retry_s = cmd_limiter.check(f"join_try:{uid}", 5, 900.0)
    if not ok_try:
        await msg.reply_text(
            f"⏳ Too many invite attempts. Please wait {int(retry_s // 60) + 1} minutes."
        )
        return

    args = list(context.args or [])
    if not args or len(args[0].strip()) < 10:
        await msg.reply_text("Usage: <code>/join &lt;invite_code&gt;</code>", parse_mode=ParseMode.HTML)
        return

    token_hash = hash_invite_token(args[0].strip())
    ok, result = db.redeem_invite_token(
        token_hash,
        uid,
        username=uname,
        display_name=display_name,
        core_topics=CORE_TOPICS,
    )
    if not ok:
        db.log_audit_event(uid, "invite_redeem_failed", result)
        await msg.reply_text(f"❌ {html.escape(result)}")
        return

    db.log_audit_event(uid, "invite_redeemed", f"role={result}")
    await msg.reply_text(
        f"✅ Welcome to <b>{html.escape(settings.org_name)}</b>! You joined as <b>{html.escape(result)}</b>.\n"
        "Run /start or vote in the poll below to personalize your AI digest.",
        parse_mode=ParseMode.HTML,
    )
    await send_onboarding_poll(context.bot, chat.id, db, user_id=uid)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return

    settings, db, llm, pipeline_lock, _, _ = _get_deps(context)
    _, _, signer = _get_security_helpers(context, settings)
    uid = int(user["user_id"])
    chat_id = update.effective_chat.id

    mode_override: str | None = None
    if context.args:
        candidate_mode = context.args[0].strip().lower()
        if candidate_mode in _VALID_DIGEST_MODES:
            mode_override = candidate_mode
        else:
            await update.effective_message.reply_text(
                "Invalid mode. Use <code>/digest balanced</code>, <code>/digest papers</code>, or <code>/digest news</code>.",
                parse_mode=ParseMode.HTML,
            )
            return

    await update.effective_message.reply_text("🔎 Checking feeds and preparing your AI digest...")

    async with pipeline_lock:
        queued_count = db.get_stats()["status_counts"]["queued"]
        last_scrape_dt = parse_iso_utc(db.get_state("last_scrape_at"))
        is_stale = (
            last_scrape_dt is None
            or (datetime.now(timezone.utc) - last_scrape_dt).total_seconds() > 300
        )
        scrape_summary: dict[str, Any] | None = None
        if queued_count == 0 or is_stale:
            scrape_summary = await run_scrape_cycle(db, llm, settings)

        result = await deliver_digest(
            context.bot,
            chat_id,
            db,
            llm,
            settings,
            user_id=uid,
            mode_override=mode_override,
            signer=signer,
        )

    if scrape_summary and scrape_summary.get("auto_disabled_feeds"):
        await notify_auto_disabled_feeds(
            context.bot, chat_id, scrape_summary["auto_disabled_feeds"]
        )

    if result["delivered"] == 0:
        await update.effective_message.reply_text(
            "No new relevant AI articles found for your profile yet. Try /discover, /trending, or /search."
        )


async def cmd_topics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    summary = db.get_user_preference_summary(uid) or db.get_state("preference_summary") or DEFAULT_PREFERENCE_SUMMARY
    rows = db.get_user_topic_weights(uid) or db.get_all_topic_weights()

    top_lines = []
    for row in rows[:12]:
        w = float(row["weight"])
        tag = "core" if row["category"] == "core" else "learned"
        top_lines.append(
            f"<code>{format_weight_bar(w)} {w:.2f}</code> "
            f"<b>{html.escape(row['topic'])}</b> ({tag}, +{row['votes']}/-{row['ignores']})"
        )

    muted_rows = [r for r in rows if float(r["weight"]) < 0.9]
    muted_lines = [
        f"• {html.escape(r['topic'])} ({float(r['weight']):.2f})" for r in muted_rows[-6:]
    ]

    parts = [
        "<b>🧠 Your AI Interest Profile</b>",
        html.escape(summary),
        "",
        "<b>Top Topic Weights:</b>",
        "\n".join(top_lines) if top_lines else "<i>No topics tracked yet.</i>",
    ]
    if muted_lines:
        parts.extend(["", "<b>Lower-Priority / Muted Topics:</b>", "\n".join(muted_lines)])

    await update.effective_message.reply_text(_join_html_lines(parts), parse_mode=ParseMode.HTML)


async def cmd_poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_chat is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    await send_onboarding_poll(
        context.bot, update.effective_chat.id, db, user_id=int(user["user_id"])
    )


async def cmd_discover(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, llm, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    recent_titles = db.get_recent_ai_titles(limit=30)
    if not recent_titles:
        await update.effective_message.reply_text(
            "No recent AI articles stored yet to discover sub-topics from. Run /digest first!"
        )
        return

    ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
    if not ok_budget:
        await update.effective_message.reply_text("⚠️ Organization daily LLM budget reached.")
        return

    sent = await send_discovery_poll(
        context.bot, update.effective_chat.id, db, llm, user_id=uid
    )
    if not sent:
        await update.effective_message.reply_text(
            "Could not find new un-tracked AI sub-topics right now. Try again after more scrapes."
        )


async def cmd_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, llm, _, _, _ = _get_deps(context)
    cmd_limiter, llm_limiter, signer = _get_security_helpers(context, settings)
    uid = int(user["user_id"])

    args = list(context.args or [])
    if not args:
        await update.effective_message.reply_text(
            "Usage: <code>/review &lt;arxiv_id | arxiv_url | library_id&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ref = " ".join(args).strip()
    paper = await _resolve_paper_ref(ref, user_id=uid, db=db, settings=settings)
    if paper is None:
        await update.effective_message.reply_text(
            "❌ Could not find or fetch that paper. Try an arXiv ID like <code>2401.00001</code>.",
            parse_mode=ParseMode.HTML,
        )
        return

    cached_json = db.get_cached_paper_review(paper["id"])
    review = None
    if cached_json:
        try:
            review = PaperReview.model_validate_json(cached_json)
        except Exception:
            review = None

    if review is None:
        ok_llm, retry_llm = llm_limiter.check(
            f"llm:{uid}", settings.rate_limit_llm_per_user_hour, 3600.0
        )
        if not ok_llm:
            await update.effective_message.reply_text(
                f"⏳ Hourly AI review quota reached. Try again in {int(retry_llm // 60) + 1} min."
            )
            return
        ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
        if not ok_budget:
            await update.effective_message.reply_text("⚠️ Organization daily LLM budget reached.")
            return

        await update.effective_message.reply_text("🔬 Analyzing paper methodology, results & limitations...")
        full_text = paper.get("full_text")
        if not full_text and paper.get("url"):
            full_text = await fetch_full_text(paper["url"])
        review = await llm.review_paper(paper, full_text=full_text)
        db.save_paper_review(paper["id"], review.model_dump_json())

    in_lib = db.get_library_item(uid, paper["id"]) is not None
    kb = build_article_action_keyboard(
        signer, uid, paper["id"], is_paper=True, in_library=in_lib
    )
    chunks = format_paper_review_html(paper, review)
    for idx, chunk in enumerate(chunks):
        kwargs: dict[str, Any] = {
            "text": chunk,
            "parse_mode": ParseMode.HTML,
            "link_preview_options": LinkPreviewOptions(is_disabled=True),
        }
        if idx == len(chunks) - 1:
            kwargs["reply_markup"] = kb
        await update.effective_message.reply_text(**kwargs)


async def cmd_litreview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context, heavy_llm=True)
    if user is None or update.effective_message is None:
        return
    settings, db, llm, _, _, _ = _get_deps(context)

    query = " ".join(context.args or []).strip()
    if len(query) < 3 or len(query) > 250:
        await update.effective_message.reply_text(
            "Usage: <code>/litreview &lt;research question, 3-250 chars&gt;</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.effective_message.reply_text(
        f"📚 Gathering grounded papers for: <i>{html.escape(query)}</i>...",
        parse_mode=ParseMode.HTML,
    )

    local_hits = db.search_articles_fts(
        query, content_type="paper", limit=6, allow_or_fallback=True
    )
    seen_ids = {p["id"] for p in local_hits}
    corpus: list[dict[str, Any]] = list(local_hits)

    if len(corpus) < 6 and settings.enable_arxiv_api:
        arxiv_hits = await search_arxiv(
            query, max_results=6, categories=settings.arxiv_categories, throttle_seconds=3.0
        )
        for rec in arxiv_hits:
            art_dict = paper_record_to_article_dict(rec)
            if art_dict["id"] not in seen_ids:
                db.upsert_paper_article(art_dict)
                saved = db.get_article(art_dict["id"])
                if saved:
                    corpus.append(saved)
                    seen_ids.add(art_dict["id"])

    if len(corpus) < 4:
        s2_hits = await search_semantic_scholar(
            query, limit=4, api_key=settings.semantic_scholar_api_key
        )
        for rec in s2_hits:
            art_dict = paper_record_to_article_dict(rec)
            if art_dict["id"] not in seen_ids:
                db.upsert_paper_article(art_dict)
                saved = db.get_article(art_dict["id"])
                if saved:
                    corpus.append(saved)
                    seen_ids.add(art_dict["id"])

    if not corpus:
        await update.effective_message.reply_text(
            "No matching papers found across your org corpus, arXiv, or Semantic Scholar."
        )
        return

    top_papers = corpus[:7]
    report = await llm.synthesize_literature_review(query, top_papers)
    chunks = format_litreview_html(query, report, top_papers)
    for chunk in chunks:
        await update.effective_message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )

    md_doc = format_litreview_markdown(query, report, top_papers)
    bib_entries = [p.get("bibtex") or format_bibtex(p) for p in top_papers]
    bib_doc = "\n\n".join(bib_entries)

    await update.effective_message.reply_document(
        document=InputFile(io.BytesIO(md_doc.encode("utf-8")), filename="litreview.md"),
        caption="📄 Markdown Literature Review",
    )
    await update.effective_message.reply_document(
        document=InputFile(io.BytesIO(bib_doc.encode("utf-8")), filename="references.bib"),
        caption="📋 BibTeX Citations",
    )


async def cmd_compare(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context, heavy_llm=True)
    if user is None or update.effective_message is None:
        return
    settings, db, llm, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    refs = list(context.args or [])
    if len(refs) < 2 or len(refs) > 4:
        await update.effective_message.reply_text(
            "Usage: <code>/compare &lt;id1&gt; &lt;id2&gt; [id3] [id4]</code> (2 to 4 arXiv/library IDs)",
            parse_mode=ParseMode.HTML,
        )
        return

    papers: list[dict[str, Any]] = []
    for ref in refs:
        resolved = await _resolve_paper_ref(ref, user_id=uid, db=db, settings=settings)
        if resolved is None:
            await update.effective_message.reply_text(
                f"❌ Could not resolve paper reference: <code>{html.escape(ref)}</code>",
                parse_mode=ParseMode.HTML,
            )
            return
        papers.append(resolved)

    report = await llm.compare_papers(papers)
    for chunk in format_paper_comparison_html(papers, report):
        await update.effective_message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)

    args = list(context.args or [])
    force_arxiv = False
    if args and args[0].lower() == "--arxiv":
        force_arxiv = True
        args = args[1:]

    query = " ".join(args).strip()
    if not query:
        await update.effective_message.reply_text(
            "Usage: <code>/search [--arxiv] &lt;query&gt;</code>", parse_mode=ParseMode.HTML
        )
        return

    local_hits = [] if force_arxiv else db.search_articles_fts(query, limit=6)
    seen_ids = {h["id"] for h in local_hits}
    results = list(local_hits)

    if (force_arxiv or len(results) < 4) and settings.enable_arxiv_api:
        arxiv_hits = await search_arxiv(
            query, max_results=5, categories=settings.arxiv_categories, throttle_seconds=3.0
        )
        for rec in arxiv_hits:
            art_dict = paper_record_to_article_dict(rec)
            if art_dict["id"] not in seen_ids:
                db.upsert_paper_article(art_dict)
                saved = db.get_article(art_dict["id"])
                if saved:
                    results.append(saved)
                    seen_ids.add(art_dict["id"])

    if not results:
        await update.effective_message.reply_text("No matching articles or papers found.")
        return

    lines = [f"<b>🔍 Search Results for: {html.escape(query[:120])}</b>"]
    for idx, item in enumerate(results[:8], start=1):
        badge = "📄" if item.get("content_type") == "paper" else "📰"
        title = html.escape(str(item.get("title") or "Untitled")[:150], quote=True)
        url = _safe_href(item.get("url"))
        ref = item.get("arxiv_id") or str(item["id"])[:10]
        lines.append(
            f'{idx}. {badge} <a href="{url}">{title}</a>\n'
            f"   <code>/review {html.escape(str(ref))}</code> · <code>/save {html.escape(str(ref))}</code>"
        )

    await update.effective_message.reply_text(
        _join_html_lines(lines),
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


async def cmd_trending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)

    hf_papers = [
        p for p in db.get_recent_papers(days=3, limit=8) if p.get("hf_upvotes") is not None
    ]
    if not hf_papers and settings.enable_hf_daily_papers:
        fetched = await fetch_hf_daily_papers(limit=10)
        if fetched:
            db.insert_new_articles(fetched)
            hf_papers = fetched[:6]

    org_trending = db.get_org_trending_articles(days=7, limit=6)
    await update.effective_message.reply_text(
        format_trending_html(hf_papers, org_trending),
        parse_mode=ParseMode.HTML,
        link_preview_options=LinkPreviewOptions(is_disabled=True),
    )


async def cmd_save(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    args = list(context.args or [])
    if not args:
        await update.effective_message.reply_text(
            "Usage: <code>/save &lt;arxiv_id|url|article_id&gt; [tag1,tag2]</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    ref = args[0].strip()
    tags = [t.strip() for t in args[1].split(",") if t.strip()] if len(args) >= 2 else []

    article = await _resolve_paper_ref(ref, user_id=uid, db=db, settings=settings)
    if article is None and is_safe_public_url(ref):
        art_id = compute_article_id(normalize_url(ref))
        db.insert_new_articles(
            [
                {
                    "id": art_id,
                    "url": ref,
                    "title": ref[:160],
                    "source_feed": "manual",
                    "rss_snippet": "",
                }
            ]
        )
        article = db.get_article(art_id)

    if article is None:
        await update.effective_message.reply_text("❌ Could not resolve item to save.")
        return

    saved = db.save_to_library(uid, article["id"], tags=tags)
    await update.effective_message.reply_text(
        f"🔖 Saved as <b>#{saved['id']}</b>: {html.escape(saved['title'][:150])}\n"
        f"Add notes with <code>/note {saved['id']} &lt;your notes&gt;</code>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_library(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    arg = context.args[0].strip() if context.args else None
    status_filter = arg if arg in {"to_read", "reading", "read", "archived"} else None
    tag_filter = None if status_filter else arg

    items = db.get_user_library(uid, status=status_filter, tag=tag_filter, limit=30)
    for chunk in format_library_html(items):
        await update.effective_message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )


async def cmd_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    args = list(context.args or [])
    if len(args) < 2:
        await update.effective_message.reply_text(
            "Usage: <code>/note &lt;lib_id&gt; &lt;note text&gt;</code>", parse_mode=ParseMode.HTML
        )
        return
    ref = args[0]
    note_text = " ".join(args[1:])
    if db.update_library_notes(uid, ref, note_text):
        await update.effective_message.reply_text(f"📝 Note saved on library item #{html.escape(ref)}.")
    else:
        await update.effective_message.reply_text("❌ Library item not found.")


async def cmd_read(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    if not context.args:
        await update.effective_message.reply_text(
            "Usage: <code>/read &lt;lib_id&gt;</code>", parse_mode=ParseMode.HTML
        )
        return
    ref = context.args[0]
    if db.update_library_status(uid, ref, "read"):
        await update.effective_message.reply_text(f"✅ Marked library item #{html.escape(ref)} as read.")
    else:
        await update.effective_message.reply_text("❌ Library item not found.")


async def cmd_bibtex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    if not context.args:
        await update.effective_message.reply_text(
            "Usage: <code>/bibtex &lt;arxiv_id|url|lib_id&gt;</code>", parse_mode=ParseMode.HTML
        )
        return
    paper = await _resolve_paper_ref(" ".join(context.args), user_id=uid, db=db, settings=settings)
    if paper is None:
        await update.effective_message.reply_text("❌ Paper or article not found.")
        return

    bib = paper.get("bibtex") or format_bibtex(paper)
    await update.effective_message.reply_text(
        f"<pre>{html.escape(bib)}</pre>", parse_mode=ParseMode.HTML
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    fmt = context.args[0].strip().lower() if context.args else "bibtex"
    if fmt not in {"bibtex", "bib", "markdown", "md"}:
        await update.effective_message.reply_text(
            "Usage: <code>/export [bibtex|markdown]</code>", parse_mode=ParseMode.HTML
        )
        return

    items = db.get_user_library(uid, limit=200)
    if not items:
        await update.effective_message.reply_text("Your library is empty. Save items first with /save.")
        return

    if fmt in {"bibtex", "bib"}:
        bib_content = "\n\n".join(it.get("bibtex") or format_bibtex(it) for it in items)
        await update.effective_message.reply_document(
            document=InputFile(io.BytesIO(bib_content.encode("utf-8")), filename="library.bib"),
            caption=f"📋 Exported {len(items)} BibTeX entries.",
        )
    else:
        md_content = format_library_markdown(items, org_name=settings.org_name)
        await update.effective_message.reply_document(
            document=InputFile(io.BytesIO(md_content.encode("utf-8")), filename="library.md"),
            caption=f"📄 Exported {len(items)} library items.",
        )


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    args = list(context.args or [])
    if not args:
        cadence = user.get("digest_interval_hours") or settings.digest_interval_hours
        max_art = user.get("max_articles_per_digest") or settings.max_articles_per_digest
        quiet = user.get("quiet_hours") or settings.quiet_hours or "off"
        mode = user.get("digest_mode") or settings.default_digest_mode
        paused = "Paused" if user.get("digests_paused") else "Active"
        await update.effective_message.reply_text(
            "<b>⚙️ Your Personal Digest Settings</b>\n"
            f"• <b>Status:</b> {paused}\n"
            f"• <b>Cadence:</b> every {cadence}h\n"
            f"• <b>Max Items:</b> {max_art} per digest\n"
            f"• <b>Mode:</b> {html.escape(mode)} (balanced | papers | news)\n"
            f"• <b>Quiet Hours:</b> {html.escape(quiet)}\n\n"
            "Update with:\n"
            "<code>/settings cadence 6</code> | <code>/settings max 5</code>\n"
            "<code>/settings mode papers</code> | <code>/settings quiet 23-07</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    sub = args[0].lower()
    if sub == "cadence" and len(args) >= 2:
        try:
            hours = float(args[1])
            if not (0.25 <= hours <= 168.0):
                raise ValueError
        except ValueError:
            await update.effective_message.reply_text("Cadence must be between 0.25 and 168 hours.")
            return
        db.update_user_settings(uid, digest_interval_hours=hours)
        await update.effective_message.reply_text(f"✅ Digest cadence updated to every {hours}h.")
        return

    if sub == "max" and len(args) >= 2:
        try:
            n = int(args[1])
            if not (1 <= n <= 9):
                raise ValueError
        except ValueError:
            await update.effective_message.reply_text("Max items must be an integer from 1 to 9.")
            return
        db.update_user_settings(uid, max_articles_per_digest=n)
        await update.effective_message.reply_text(f"✅ Max digest items updated to {n}.")
        return

    if sub == "mode" and len(args) >= 2:
        mode = args[1].lower()
        if mode not in _VALID_DIGEST_MODES:
            await update.effective_message.reply_text("Mode must be: balanced, papers, or news.")
            return
        db.update_user_settings(uid, digest_mode=mode)
        await update.effective_message.reply_text(f"✅ Digest mode updated to {mode}.")
        return

    if sub == "quiet" and len(args) >= 2:
        val = args[1].strip().lower()
        if val in {"off", "none", "disable"}:
            db.update_user_settings(uid, quiet_hours=None)
            await update.effective_message.reply_text("✅ Quiet hours disabled.")
            return
        match = QUIET_HOURS_RE.fullmatch(val)
        if not match or int(match.group(1)) == int(match.group(2)):
            await update.effective_message.reply_text("Use HH-HH format (e.g. 23-07) or 'off'.")
            return
        formatted = f"{int(match.group(1)):02d}-{int(match.group(2)):02d}"
        db.update_user_settings(uid, quiet_hours=formatted)
        await update.effective_message.reply_text(f"✅ Quiet hours updated to {formatted}.")
        return

    await update.effective_message.reply_text(
        "Usage: <code>/settings [cadence H | max 1-9 | mode balanced|papers|news | quiet HH-HH|off]</code>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_feeds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])

    args = list(context.args or [])
    if not args:
        feeds = db.get_enabled_feeds()
        if not feeds:
            await update.effective_message.reply_text(
                "No enabled feeds. Add one with <code>/feeds add &lt;url&gt;</code>.",
                parse_mode=ParseMode.HTML,
            )
            return
        lines = ["<b>📡 Enabled RSS Feeds:</b>"]
        for idx, feed in enumerate(feeds, start=1):
            checked = feed.get("last_checked_at") or "never"
            failures = int(feed.get("consecutive_failures") or 0)
            warn = ""
            if failures > 0:
                err = html.escape(str(feed.get("last_error") or "error")[:80])
                warn = f"\n   ⚠️ failing ({failures}x): {err}"
            lines.append(
                f"{idx}. <code>{html.escape(feed['url'])}</code> (checked: {html.escape(checked)}){warn}"
            )
        if user["role"] == "admin":
            lines.append(
                "\nManage: <code>/feeds add &lt;url&gt;</code> or <code>/feeds remove &lt;number|url&gt;</code>"
            )
        await update.effective_message.reply_text(_join_html_lines(lines), parse_mode=ParseMode.HTML)
        return

    if user["role"] != "admin":
        db.log_audit_event(uid, "rbac_denied", f"/feeds {args[0]}")
        await update.effective_message.reply_text("🚫 Only organization admins can modify RSS feeds.")
        return

    subcmd = args[0].lower()
    if subcmd == "add" and len(args) >= 2:
        raw_url = args[1].strip()
        if not is_safe_public_url(raw_url):
            await update.effective_message.reply_text(
                "Invalid or unsafe URL. Please supply a public http:// or https:// feed URL."
            )
            return
        is_valid, feed_title, err = await validate_rss_feed_url(raw_url)
        if not is_valid:
            await update.effective_message.reply_text(f"❌ Feed rejected: {err}")
            return
        changed = db.add_feed(raw_url, title=feed_title)
        db.log_audit_event(uid, "feed_added", raw_url)
        msg = f"✅ Feed added/enabled: {raw_url}" if changed else f"ℹ️ Feed already active: {raw_url}"
        await update.effective_message.reply_text(msg)
        return

    if subcmd == "remove" and len(args) >= 2:
        target = args[1].strip()
        if target.isdigit():
            idx = int(target)
            enabled = db.get_enabled_feeds()
            if not (1 <= idx <= len(enabled)):
                await update.effective_message.reply_text(
                    f"Feed index {idx} out of range (1..{len(enabled)})."
                )
                return
            target_url = enabled[idx - 1]["url"]
        else:
            target_url = target

        removed = db.remove_feed(target_url)
        if removed:
            db.log_audit_event(uid, "feed_removed", target_url)
        msg = (
            f"🗑️ Feed disabled: {target_url}"
            if removed
            else f"⚠️ Enabled feed not found: {target_url}"
        )
        await update.effective_message.reply_text(msg)
        return

    await update.effective_message.reply_text(
        "Usage: <code>/feeds</code>, <code>/feeds add &lt;url&gt;</code>, or <code>/feeds remove &lt;number|url&gt;</code>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)

    stats = db.get_stats()
    counts = stats["status_counts"]
    last_scrape = stats["last_scrape_at"] or "never"
    user_paused = bool(user.get("digests_paused")) or stats["digests_paused"]
    paused_label = "⏸️ Paused" if user_paused else "▶️ Active"
    quiet_label = user.get("quiet_hours") or settings.quiet_hours or "disabled"
    auto_disable_label = (
        f"{settings.feed_max_consecutive_failures} failures"
        if settings.feed_max_consecutive_failures > 0
        else "off"
    )
    retention_label = (
        f"{settings.article_retention_days}d"
        if settings.article_retention_days > 0
        else "off"
    )

    text = (
        f"<b>📊 {html.escape(settings.org_name)} Status</b>\n"
        f"• <b>Feeds:</b> {stats['enabled_feeds']} enabled ({stats['failing_feeds']} failing)\n"
        f"• <b>Corpus:</b> papers={stats['paper_count']}, queued={counts['queued']}, "
        f"new={counts['new']}, filtered={counts['filtered_out']}, "
        f"expired={counts['expired']}, failed={counts['failed']}, delivered={counts['delivered']}\n"
        f"• <b>Org Users:</b> {stats['active_users']} active ({html.escape(user['role'])})\n"
        f"• <b>LLM Budget Today:</b> {stats['daily_llm_usage']} / {settings.daily_org_llm_budget}\n"
        f"• <b>Last Scrape:</b> {html.escape(last_scrape)}\n"
        f"• <b>Your Digests:</b> {int(user.get('digests_delivered_count') or stats['digests_delivered_count'])} ({paused_label}, quiet: {html.escape(quiet_label)})\n"
        f"• <b>LLM Model:</b> <code>{html.escape(settings.llm_model)}</code> ({html.escape(settings.llm_provider)})\n"
        f"• <b>Intervals:</b> scrape={settings.scrape_interval_minutes}m, "
        f"digest={user.get('digest_interval_hours') or settings.digest_interval_hours}h\n"
        f"• <b>Lifecycle:</b> auto-disable={auto_disable_label}, retention={retention_label}"
    )
    await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])
    db.update_user_settings(uid, digests_paused=1)
    if db.count_active_users() <= 1:
        db.set_state("digests_paused", "1")
    await update.effective_message.reply_text(
        "⏸️ Scheduled digests paused for you. You can still run /digest manually anytime or /resume later."
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context)
    if user is None or update.effective_message is None:
        return
    _, db, _, _, _, _ = _get_deps(context)
    uid = int(user["user_id"])
    db.update_user_settings(uid, digests_paused=0)
    db.set_state("digests_paused", "0")
    await update.effective_message.reply_text("▶️ Scheduled digests resumed.")


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await _require_auth_and_limits(update, context, min_role="admin")
    if user is None or update.effective_message is None:
        return
    settings, db, _, _, _, _ = _get_deps(context)
    actor_id = int(user["user_id"])

    args = list(context.args or [])
    if not args:
        await update.effective_message.reply_text(
            "<b>🛡️ Admin Console</b>\n"
            "• <code>/admin invite [member|admin] [max_uses] [hours]</code>\n"
            "• <code>/admin users</code> — List org users\n"
            "• <code>/admin approve &lt;user_id&gt;</code>\n"
            "• <code>/admin promote &lt;user_id&gt;</code> | <code>demote &lt;user_id&gt;</code>\n"
            "• <code>/admin revoke &lt;user_id&gt;</code> | <code>revoke_invites</code>\n"
            "• <code>/admin audit</code> | <code>budget</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    sub = args[0].lower()

    if sub == "invite":
        role = "member"
        max_uses = 1
        hours = 72
        if len(args) >= 2 and args[1].lower() in {"member", "admin"}:
            role = args[1].lower()
            idx_offset = 2
        else:
            idx_offset = 1
        if len(args) > idx_offset and args[idx_offset].isdigit():
            max_uses = max(1, min(100, int(args[idx_offset])))
        if len(args) > idx_offset + 1 and args[idx_offset + 1].isdigit():
            hours = max(1, min(720, int(args[idx_offset + 1])))

        raw_token = generate_invite_token()
        db.create_invite_token(
            hash_invite_token(raw_token),
            created_by=actor_id,
            role=role,
            max_uses=max_uses,
            expires_hours=hours,
        )
        db.log_audit_event(
            actor_id, "invite_created", f"role={role},max_uses={max_uses},hours={hours}"
        )
        await update.effective_message.reply_text(
            f"🎟️ <b>Invite Created</b> (role: <b>{role}</b>, uses: {max_uses}, expires: {hours}h)\n\n"
            f"Share this command with your colleague in a 1:1 DM with the bot:\n"
            f"<code>/join {html.escape(raw_token)}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    if sub == "users":
        users = db.list_users()
        lines = [f"<b>👥 Organization Users ({len(users)})</b>"]
        for u in users[:40]:
            uname = f"@{u['username']}" if u.get("username") else (u.get("display_name") or "User")
            lines.append(
                f"• <code>{u['user_id']}</code> {html.escape(uname)} — "
                f"<b>{html.escape(u['role'])}</b> [{html.escape(u['status'])}]"
            )
        await update.effective_message.reply_text(_join_html_lines(lines), parse_mode=ParseMode.HTML)
        return

    if sub in {"approve", "promote", "demote", "revoke"} and len(args) >= 2:
        try:
            target_id = int(args[1])
        except ValueError:
            await update.effective_message.reply_text("❌ Target user_id must be a numeric Telegram ID.")
            return

        target = db.get_user(target_id)
        if sub == "approve":
            if target is None:
                db.upsert_user(target_id, role="member", status="active", core_topics=CORE_TOPICS)
            else:
                db.set_user_status(target_id, "active")
                db.seed_user_core_topics(target_id, CORE_TOPICS)
            db.log_audit_event(actor_id, "user_approved", str(target_id))
            await update.effective_message.reply_text(f"✅ Approved user <code>{target_id}</code>.", parse_mode=ParseMode.HTML)
            return

        if target is None:
            await update.effective_message.reply_text("❌ User not found.")
            return

        if sub == "promote":
            db.set_user_role(target_id, "admin")
            db.set_user_status(target_id, "active")
            db.log_audit_event(actor_id, "user_promoted", str(target_id))
            await update.effective_message.reply_text(f"🛡️ Promoted <code>{target_id}</code> to admin.", parse_mode=ParseMode.HTML)
            return

        if sub == "demote":
            if target["role"] == "admin" and db.count_admins() <= 1:
                await update.effective_message.reply_text("🚫 Cannot demote the last active admin.")
                return
            db.set_user_role(target_id, "member")
            db.log_audit_event(actor_id, "user_demoted", str(target_id))
            await update.effective_message.reply_text(f"👤 Demoted <code>{target_id}</code> to member.", parse_mode=ParseMode.HTML)
            return

        if sub == "revoke":
            if target["role"] == "admin" and target["status"] == "active" and db.count_admins() <= 1:
                await update.effective_message.reply_text("🚫 Cannot revoke the last active admin.")
                return
            db.set_user_status(target_id, "revoked")
            db.log_audit_event(actor_id, "user_revoked", str(target_id))
            await update.effective_message.reply_text(f"🔒 Revoked access for <code>{target_id}</code>.", parse_mode=ParseMode.HTML)
            return

    if sub == "revoke_invites":
        cnt = db.revoke_all_invites()
        db.log_audit_event(actor_id, "invites_revoked_all", str(cnt))
        await update.effective_message.reply_text(f"🔒 Revoked {cnt} active invite code(s).")
        return

    if sub == "audit":
        events = db.get_recent_audit_events(limit=20)
        lines = ["<b>📜 Recent Security Audit Log</b>"]
        for ev in events:
            lines.append(
                f"• <code>{html.escape(ev['created_at'][:19])}</code> "
                f"actor={ev['actor_user_id']} <b>{html.escape(ev['event_type'])}</b> "
                f"{html.escape(str(ev.get('target') or ''))}"
            )
        await update.effective_message.reply_text(_join_html_lines(lines), parse_mode=ParseMode.HTML)
        return

    if sub == "budget":
        used = db.get_daily_llm_usage()
        await update.effective_message.reply_text(
            f"💰 Daily Org LLM Budget: <b>{used} / {settings.daily_org_llm_budget}</b> calls used today.",
            parse_mode=ParseMode.HTML,
        )
        return

    await update.effective_message.reply_text("❌ Unknown /admin subcommand. Run /admin for usage.")


# ------------------------------------------------------------------
# Interactive Signed Callback & Poll Handlers
# ------------------------------------------------------------------


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = getattr(update, "callback_query", None)
    if query is None or query.from_user is None:
        return

    settings, db, llm, _, _, _ = _get_deps(context)
    cmd_limiter, llm_limiter, signer = _get_security_helpers(context, settings)
    uid = query.from_user.id

    user = db.get_user(uid)
    if user is None or user["status"] != "active":
        await query.answer("Unauthorized.", show_alert=True)
        return

    ok_cmd, _ = cmd_limiter.check(f"cb:{uid}", settings.rate_limit_commands_per_min, 60.0)
    if not ok_cmd:
        await query.answer("Rate limit reached. Slow down.", show_alert=True)
        return

    verified = signer.verify(uid, query.data)
    if verified is None:
        db.log_audit_event(uid, "invalid_callback_signature", str(query.data)[:60])
        await query.answer("Invalid or expired action button.", show_alert=True)
        return

    action, ref = verified
    article = db.get_article(ref)
    if article is None:
        await query.answer("Article no longer available.", show_alert=True)
        return

    if action == "sv":
        saved = db.save_to_library(uid, article["id"])
        await query.answer(f"Saved as #{saved['id']}!")
        return

    if action == "rm":
        db.remove_from_library(uid, article["id"])
        await query.answer("Removed from library.")
        return

    if action == "rd":
        db.update_library_status(uid, article["id"], "read")
        await query.answer("Marked as read!")
        return

    if action == "bx":
        bib = article.get("bibtex") or format_bibtex(article)
        await query.answer("Sent BibTeX!")
        if query.message:
            await query.message.reply_text(
                f"<pre>{html.escape(bib)}</pre>", parse_mode=ParseMode.HTML
            )
        return

    if action == "rv":
        cached_json = db.get_cached_paper_review(article["id"])
        review = None
        if cached_json:
            try:
                review = PaperReview.model_validate_json(cached_json)
            except Exception:
                review = None

        if review is None:
            ok_llm, _ = llm_limiter.check(
                f"llm:{uid}", settings.rate_limit_llm_per_user_hour, 3600.0
            )
            if not ok_llm:
                await query.answer("Hourly LLM quota reached.", show_alert=True)
                return
            ok_budget, _ = db.consume_daily_llm_budget(settings.daily_org_llm_budget, 1)
            if not ok_budget:
                await query.answer("Org daily LLM budget reached.", show_alert=True)
                return
            await query.answer("Running paper review...")
            full_text = article.get("full_text")
            if not full_text and article.get("url"):
                full_text = await fetch_full_text(article["url"])
            review = await llm.review_paper(article, full_text=full_text)
            db.save_paper_review(article["id"], review.model_dump_json())
        else:
            await query.answer("Loaded cached review!")

        if query.message:
            for chunk in format_paper_review_html(article, review):
                await query.message.reply_text(
                    chunk,
                    parse_mode=ParseMode.HTML,
                    link_preview_options=LinkPreviewOptions(is_disabled=True),
                )


async def handle_poll_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, llm, _, profile_lock, background_tasks = _get_deps(context)
    if not is_authorized(update, settings, db):
        return

    poll_answer = update.poll_answer
    if poll_answer is None:
        return

    poll = db.get_poll(poll_answer.poll_id)
    if poll is None:
        return

    option_ids = list(poll_answer.option_ids or [])
    if not option_ids:
        return

    changed, boosted, should_update_profile = process_poll_answer(db, poll, option_ids)
    if not changed:
        return

    poll_user_id: int | None = poll.get("user_id")
    if should_update_profile:
        _track_background_task(
            background_tasks,
            _refresh_profile_summary(db, llm, profile_lock, user_id=poll_user_id),
        )

    if boosted:
        text = "✅ Preferences updated! Boosted: " + ", ".join(boosted[:8])
    else:
        text = "✅ Feedback recorded! Topic weights adjusted."
    if poll["poll_type"] == "onboarding":
        text += "\nTip: send /digest anytime to get your first briefing right away!"
    await context.bot.send_message(chat_id=poll["chat_id"], text=text)


# ------------------------------------------------------------------
# Scheduled Background Jobs
# ------------------------------------------------------------------


async def scrape_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, llm, pipeline_lock, _, _ = _get_deps(context)
    if pipeline_lock.locked():
        logger.info("Skipping scheduled scrape_job because pipeline_lock is held")
        return
    summary: dict[str, Any] | None = None
    async with pipeline_lock:
        try:
            summary = await run_scrape_cycle(db, llm, settings)
            logger.info("Scrape cycle complete: %s", summary)
        except Exception as exc:
            logger.exception("Scheduled scrape_job failed: %s", exc)

    if summary and summary.get("auto_disabled_feeds"):
        admins = [u["user_id"] for u in db.list_users(status="active") if u["role"] == "admin"]
        if not admins:
            fallback = settings.telegram_allowed_chat_id or db.get_state("active_chat_id")
            if fallback is not None:
                admins = [int(fallback)]
        for admin_chat_id in admins:
            await notify_auto_disabled_feeds(
                context.bot, int(admin_chat_id), summary["auto_disabled_feeds"]
            )


async def hf_papers_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, llm, pipeline_lock, _, _ = _get_deps(context)
    if not settings.enable_hf_daily_papers or pipeline_lock.locked():
        return
    async with pipeline_lock:
        try:
            summary = await run_hf_papers_cycle(db, llm, settings)
            logger.info("HF Daily Papers sync complete: %s", summary)
        except Exception as exc:
            logger.exception("Scheduled hf_papers_job failed: %s", exc)


async def retention_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, _, pipeline_lock, _, _ = _get_deps(context)
    if settings.article_retention_days <= 0:
        return
    if pipeline_lock.locked():
        logger.info("Skipping scheduled retention_job because pipeline_lock is held")
        return
    async with pipeline_lock:
        try:
            pruned = db.prune_old_articles(
                settings.article_retention_days,
                min_age_hours=settings.max_article_age_hours,
            )
            db.set_state("last_prune_at", utc_now_iso())
            logger.info("Retention pruning complete: pruned=%d", pruned)
        except Exception as exc:
            logger.exception("Scheduled retention_job failed: %s", exc)


async def digest_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings, db, llm, pipeline_lock, _, _ = _get_deps(context)
    _, _, signer = _get_security_helpers(context, settings)

    if db.get_state("digests_paused", "0") == "1":
        logger.info("Skipping scheduled digest_job because digests are paused")
        return

    due_users = db.get_due_digest_users(settings.digest_interval_hours)
    if due_users:
        async with pipeline_lock:
            for u in due_users:
                user_quiet = u.get("quiet_hours") or settings.quiet_hours
                if is_within_quiet_hours(user_quiet):
                    continue
                uid = int(u["user_id"])
                try:
                    result = await deliver_digest(
                        context.bot,
                        uid,
                        db,
                        llm,
                        settings,
                        user_id=uid,
                        signer=signer,
                    )
                    logger.info("Scheduled digest delivered to user=%s: %s", uid, result)
                except Exception as exc:
                    logger.exception("Scheduled digest failed for user=%s: %s", uid, exc)
        return

    # Legacy single-chat fallback when no rows exist in users table
    if is_within_quiet_hours(settings.quiet_hours):
        logger.info("Skipping scheduled digest_job inside quiet hours (%s)", settings.quiet_hours)
        return
    target_chat = settings.telegram_allowed_chat_id or db.get_state("active_chat_id")
    if target_chat is None:
        return
    chat_id = int(target_chat)

    async with pipeline_lock:
        try:
            result = await deliver_digest(context.bot, chat_id, db, llm, settings)
            logger.info("Scheduled digest_job complete: %s", result)
        except Exception as exc:
            logger.exception("Scheduled digest_job failed: %s", exc)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled exceptions from Telegram handlers without crashing the bot."""
    logger.exception("Unhandled Telegram handler exception", exc_info=context.error)


async def _post_init(application: Application) -> None:
    commands = [
        BotCommand("start", "Onboard & run core topic poll"),
        BotCommand("digest", "Personalized AI & paper digest now"),
        BotCommand("review", "Structured AI peer review of a paper"),
        BotCommand("litreview", "Multi-paper literature review synthesis"),
        BotCommand("compare", "Compare 2-4 papers side-by-side"),
        BotCommand("search", "Search org corpus & arXiv papers"),
        BotCommand("trending", "HF Daily Papers & org top picks"),
        BotCommand("library", "View your saved reading library"),
        BotCommand("save", "Save a paper or article to library"),
        BotCommand("bibtex", "Get BibTeX citation"),
        BotCommand("export", "Export library as BibTeX or Markdown"),
        BotCommand("topics", "Show preference summary & weights"),
        BotCommand("poll", "Re-run the 8-topic core preference poll"),
        BotCommand("discover", "Discover emerging AI sub-topics"),
        BotCommand("settings", "Configure personal digest preferences"),
        BotCommand("feeds", "List or manage RSS feeds"),
        BotCommand("status", "Show corpus health, budget & schedule"),
        BotCommand("pause", "Pause scheduled digests"),
        BotCommand("resume", "Resume scheduled digests"),
        BotCommand("help", "Full command reference"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except Exception as exc:
        logger.warning("Failed setting Telegram bot commands: %s", exc)


async def _post_shutdown(application: Application) -> None:
    tasks: set[asyncio.Task[Any]] = application.bot_data.get("background_tasks", set())
    for task in list(tasks):
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    llm: LLMClient | None = application.bot_data.get("llm")
    if llm is not None:
        await llm.close()


def build_application(
    settings: Settings, db: Database, llm: LLMClient
) -> Application:
    """Construct and wire the Telegram Application, handlers, rate limiter, and JobQueue."""
    rate_limiter = TelegramRateLimiter()
    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token)
        .rate_limiter(rate_limiter)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    application.bot_data["settings"] = settings
    application.bot_data["db"] = db
    application.bot_data["llm"] = llm
    application.bot_data["pipeline_lock"] = asyncio.Lock()
    application.bot_data["profile_lock"] = asyncio.Lock()
    application.bot_data["background_tasks"] = set()
    application.bot_data["command_limiter"] = SlidingWindowRateLimiter()
    application.bot_data["llm_limiter"] = SlidingWindowRateLimiter()
    application.bot_data["callback_signer"] = CallbackSigner(settings.telegram_bot_token)

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("join", cmd_join))
    application.add_handler(CommandHandler("digest", cmd_digest))
    application.add_handler(CommandHandler("review", cmd_review))
    application.add_handler(CommandHandler("litreview", cmd_litreview))
    application.add_handler(CommandHandler("compare", cmd_compare))
    application.add_handler(CommandHandler("search", cmd_search))
    application.add_handler(CommandHandler("trending", cmd_trending))
    application.add_handler(CommandHandler("library", cmd_library))
    application.add_handler(CommandHandler("save", cmd_save))
    application.add_handler(CommandHandler("note", cmd_note))
    application.add_handler(CommandHandler("read", cmd_read))
    application.add_handler(CommandHandler("bibtex", cmd_bibtex))
    application.add_handler(CommandHandler("export", cmd_export))
    application.add_handler(CommandHandler("topics", cmd_topics))
    application.add_handler(CommandHandler("poll", cmd_poll))
    application.add_handler(CommandHandler("discover", cmd_discover))
    application.add_handler(CommandHandler("settings", cmd_settings))
    application.add_handler(CommandHandler("feeds", cmd_feeds))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("pause", cmd_pause))
    application.add_handler(CommandHandler("resume", cmd_resume))
    application.add_handler(CommandHandler("admin", cmd_admin))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(PollAnswerHandler(handle_poll_answer))
    application.add_error_handler(error_handler)

    if application.job_queue is not None:
        application.job_queue.run_repeating(
            scrape_job,
            interval=settings.scrape_interval_minutes * 60,
            first=10,
            name="scrape_job",
        )
        if settings.enable_hf_daily_papers:
            application.job_queue.run_repeating(
                hf_papers_job,
                interval=max(1800, settings.scrape_interval_minutes * 60),
                first=25,
                name="hf_papers_job",
            )
        tick_seconds = settings.digest_tick_minutes * 60
        application.job_queue.run_repeating(
            digest_job,
            interval=tick_seconds,
            first=tick_seconds,
            name="digest_job",
        )
        if settings.article_retention_days > 0:
            application.job_queue.run_repeating(
                retention_job,
                interval=24 * 3600,
                first=60,
                name="retention_job",
            )

    return application
