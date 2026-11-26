"""Hybrid poll preference learning math, per-user ranking, and poll builders."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from db import Database, parse_iso_utc
from llm import ProfileContext

CORE_TOPICS = [
    "LLMs & Foundation Models",
    "AI Agents & Tool Use",
    "AI Coding & DevTools",
    "Open-Source & Local Models",
    "AI Research & Architectures",
    "RAG, Memory & Data Engineering",
    "AI Hardware, Chips & Inference",
    "AI Startups, Product & Policy",
]

MIN_WEIGHT = 0.1
MAX_WEIGHT = 3.0
DEFAULT_WEIGHT = 1.0

DEFAULT_PREFERENCE_SUMMARY = (
    "Technical AI researcher and engineer interested in foundation models, practical AI agents, "
    "developer tooling, open-source releases, and high-signal ML papers."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "with"}


def clamp_weight(weight: float) -> float:
    """Clamp topic weight into [MIN_WEIGHT, MAX_WEIGHT] rounded to 4 decimals."""
    return round(max(MIN_WEIGHT, min(MAX_WEIGHT, float(weight))), 4)


def resolve_canonical_topic(
    raw_topic: str, existing_weights: dict[str, dict[str, Any]]
) -> str:
    """Resolve raw topic string against existing topic keys case-insensitively."""
    cleaned = " ".join(str(raw_topic).strip().split())
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for existing_key in existing_weights:
        if existing_key.lower() == lowered:
            return existing_key
    return cleaned


def compute_blended_score(
    llm_score: float,
    topics: list[str],
    topic_weights: dict[str, dict[str, Any]],
) -> float:
    """Compute neutral-centered score adjustment over matched known topics only."""
    matched_weights: list[float] = []
    seen_canonical: set[str] = set()
    for raw_topic in topics:
        canonical = resolve_canonical_topic(raw_topic, topic_weights)
        if not canonical or canonical in seen_canonical:
            continue
        if canonical in topic_weights:
            seen_canonical.add(canonical)
            matched_weights.append(float(topic_weights[canonical]["weight"]))

    if matched_weights:
        w_bar = sum(matched_weights) / len(matched_weights)
    else:
        w_bar = DEFAULT_WEIGHT

    if w_bar >= 1.0:
        delta = 0.20 * ((w_bar - 1.0) / 2.0)
    else:
        delta = -0.20 * ((1.0 - w_bar) / 0.9)

    s_final = max(0.0, min(1.0, float(llm_score) + delta))
    return round(s_final, 4)


def _tokenize_title(title: str) -> set[str]:
    tokens = _TOKEN_RE.findall(title.lower())
    filtered = {tok for tok in tokens if tok not in _STOPWORDS}
    return filtered if filtered else set(tokens)


def title_jaccard_similarity(title_a: str, title_b: str) -> float:
    """Compute token Jaccard similarity between two article titles."""
    set_a = _tokenize_title(title_a)
    set_b = _tokenize_title(title_b)
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def select_digest_articles(
    candidates: list[dict[str, Any]],
    max_articles: int,
    *,
    max_age_hours: int = 72,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Select diverse Top-K digest candidates using recency decay and title dedup."""
    if not candidates or max_articles <= 0:
        return []

    ref_now = now or datetime.now(timezone.utc)
    scored_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for item in candidates:
        ts = parse_iso_utc(item.get("published_at")) or parse_iso_utc(item.get("created_at"))
        if ts is not None:
            age_hours = max(0.0, min(float(max_age_hours), (ref_now - ts).total_seconds() / 3600.0))
        else:
            age_hours = 0.0
        base_score = float(item.get("relevance_score") or 0.0)
        rank_score = base_score - (0.0025 * age_hours)
        sort_ts = item.get("published_at") or item.get("created_at") or ""
        scored_candidates.append((rank_score, sort_ts, item))

    scored_candidates.sort(key=lambda tup: (tup[0], tup[1]), reverse=True)

    selected: list[dict[str, Any]] = []
    for _, _, item in scored_candidates:
        title = item.get("title") or ""
        if any(title_jaccard_similarity(title, picked.get("title") or "") > 0.55 for picked in selected):
            continue
        selected.append(item)
        if len(selected) >= max_articles:
            break
    return selected


def _decode_topics_json(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw if str(t).strip()]
    if not raw:
        return []
    try:
        loaded = json.loads(str(raw))
        if isinstance(loaded, list):
            return [str(t) for t in loaded if str(t).strip()]
    except (TypeError, ValueError):
        pass
    return []


def rank_candidates_for_user(
    candidates: list[dict[str, Any]],
    user_weights: dict[str, dict[str, Any]],
    max_articles: int,
    *,
    min_relevance_score: float = 0.65,
    max_age_hours: int = 72,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Personalize candidate relevance scores for a specific user and select diverse Top-K."""
    personalized: list[dict[str, Any]] = []
    for item in candidates:
        base_llm = float(item.get("llm_score") if item.get("llm_score") is not None else item.get("relevance_score") or 0.7)
        topics = _decode_topics_json(item.get("topics_json"))
        personal_score = compute_blended_score(base_llm, topics, user_weights)

        # Mild boost for high-traction papers (HF upvotes / citations)
        hf_upvotes = int(item.get("hf_upvotes") or 0)
        citations = int(item.get("citation_count") or 0)
        signal_bonus = 0.0
        if hf_upvotes >= 20 or citations >= 50:
            signal_bonus = 0.05
        elif hf_upvotes >= 5 or citations >= 10:
            signal_bonus = 0.02
        final_score = round(min(1.0, personal_score + signal_bonus), 4)

        if final_score < min_relevance_score:
            continue
        enriched_copy = dict(item)
        enriched_copy["relevance_score"] = final_score
        personalized.append(enriched_copy)

    return select_digest_articles(
        personalized,
        max_articles,
        max_age_hours=max_age_hours,
        now=now,
    )


def process_poll_answer(
    db: Database, poll: dict[str, Any], option_ids: list[int]
) -> tuple[bool, list[str], bool]:
    """Idempotently process a Telegram PollAnswer update (per-user or legacy global)."""
    if not option_ids:
        return (False, [], False)

    if not db.claim_poll_answer(poll["poll_id"], list(option_ids)):
        return (False, [], False)

    options_metadata = poll.get("options_metadata")
    if options_metadata is None:
        options_metadata = json.loads(poll["options_metadata_json"])

    user_id: int | None = poll.get("user_id")
    if user_id is not None:
        existing_rows = db.get_user_topic_weights(int(user_id))
    else:
        existing_rows = db.get_all_topic_weights()
    existing_map: dict[str, dict[str, Any]] = {row["topic"]: dict(row) for row in existing_rows}

    def _upsert_weight(
        topic: str,
        category: str,
        new_w: float,
        vote_delta: int = 0,
        ignore_delta: int = 0,
    ) -> None:
        if user_id is not None:
            db.upsert_user_topic_weight(
                int(user_id), topic, category, new_w, vote_delta=vote_delta, ignore_delta=ignore_delta
            )
        else:
            db.upsert_topic_weight(
                topic, category, new_w, vote_delta=vote_delta, ignore_delta=ignore_delta
            )

    poll_type = poll["poll_type"]
    boosted: list[str] = []
    option_set = set(option_ids)

    if poll_type == "onboarding":
        for idx, meta in enumerate(options_metadata):
            raw_topic = meta.get("topic")
            if not raw_topic:
                continue
            canonical = resolve_canonical_topic(raw_topic, existing_map)
            current = float(existing_map.get(canonical, {}).get("weight", DEFAULT_WEIGHT))
            category = existing_map.get(canonical, {}).get("category", "core")
            if idx in option_set:
                new_w = clamp_weight(max(current, 1.5) + 0.3)
                _upsert_weight(canonical, category, new_w, vote_delta=1)
                existing_map[canonical] = {"topic": canonical, "category": category, "weight": new_w}
                boosted.append(canonical)
            else:
                new_w = clamp_weight(min(current, 0.7))
                _upsert_weight(canonical, category, new_w, vote_delta=0)
                existing_map[canonical] = {"topic": canonical, "category": category, "weight": new_w}
        return (True, boosted, True)

    if poll_type == "digest_feedback":
        voted_indices = {
            idx
            for idx in option_ids
            if 0 <= idx < len(options_metadata) and not options_metadata[idx].get("none_option")
        }

        liked_ids: list[str] = []
        skipped_ids: list[str] = []
        t_voted: list[str] = []
        t_voted_seen: set[str] = set()
        t_unvoted_candidates: list[str] = []
        t_unvoted_seen: set[str] = set()

        for idx, meta in enumerate(options_metadata):
            if meta.get("none_option"):
                continue
            article_id = meta.get("article_id")
            raw_topics = meta.get("topics") or []
            canon_topics: list[str] = []
            for raw in raw_topics:
                canon = resolve_canonical_topic(raw, existing_map)
                if canon:
                    canon_topics.append(canon)

            if idx in voted_indices:
                if article_id:
                    liked_ids.append(article_id)
                for canon in canon_topics:
                    if canon not in t_voted_seen:
                        t_voted_seen.add(canon)
                        t_voted.append(canon)
            else:
                if article_id:
                    skipped_ids.append(article_id)
                for canon in canon_topics:
                    if canon not in t_unvoted_seen:
                        t_unvoted_seen.add(canon)
                        t_unvoted_candidates.append(canon)

        t_unvoted = [t for t in t_unvoted_candidates if t not in t_voted_seen]

        for topic in t_voted:
            current = float(existing_map.get(topic, {}).get("weight", DEFAULT_WEIGHT))
            category = existing_map.get(topic, {}).get("category", "learned")
            new_w = clamp_weight(current + 0.25)
            _upsert_weight(topic, category, new_w, vote_delta=1)
            existing_map[topic] = {"topic": topic, "category": category, "weight": new_w}
            boosted.append(topic)

        for topic in t_unvoted:
            if topic not in existing_map:
                continue
            current = float(existing_map[topic]["weight"])
            category = existing_map[topic]["category"]
            new_w = clamp_weight(current - 0.08)
            _upsert_weight(topic, category, new_w, ignore_delta=1)
            existing_map[topic]["weight"] = new_w

        if user_id is not None:
            db.set_user_articles_feedback(int(user_id), liked_ids, skipped_ids)
        else:
            db.set_articles_feedback(liked_ids, skipped_ids)
        return (True, boosted, True)

    if poll_type == "topic_discovery":
        for idx in option_ids:
            if not (0 <= idx < len(options_metadata)):
                continue
            meta = options_metadata[idx]
            if meta.get("none_option"):
                continue
            raw_topic = meta.get("topic")
            if not raw_topic:
                continue
            canonical = resolve_canonical_topic(raw_topic, existing_map)
            current = float(existing_map.get(canonical, {}).get("weight", DEFAULT_WEIGHT))
            category = existing_map.get(canonical, {}).get("category", "learned")
            new_w = clamp_weight(max(current, 1.5))
            _upsert_weight(canonical, category, new_w, vote_delta=1)
            existing_map[canonical] = {"topic": canonical, "category": category, "weight": new_w}
            if canonical not in boosted:
                boosted.append(canonical)
        return (True, boosted, len(boosted) > 0)

    return (False, [], False)


def _truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    cut = text[: max_len - 3].rstrip()
    return f"{cut}..."


def build_safe_poll_options(
    items: list[tuple[str, dict[str, Any]]],
    sentinel_label: str = "None of these / More variety",
    max_options: int = 10,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build Telegram-compliant multi-select poll options and aligned metadata."""
    if not items:
        return ([], [])

    effective_max = max(2, min(10, int(max_options)))
    primary_items = items[: effective_max - 1]

    clean_sentinel = " ".join(str(sentinel_label or "").split()) or "None of these"
    clean_sentinel = _truncate_text(clean_sentinel, 100)

    options: list[str] = []
    metadata_list: list[dict[str, Any]] = []
    used: set[str] = set()

    for raw_label, meta in primary_items:
        base = " ".join(str(raw_label or "").split()) or "Untitled item"
        candidate = _truncate_text(base, 100)
        if candidate in used or candidate == clean_sentinel:
            counter = 2
            while True:
                suffix = f" ({counter})"
                budget = 100 - len(suffix)
                prefixed = f"{_truncate_text(base, budget)}{suffix}"
                if prefixed not in used and prefixed != clean_sentinel:
                    candidate = prefixed
                    break
                counter += 1
        used.add(candidate)
        options.append(candidate)
        metadata_list.append(dict(meta))

    options.append(clean_sentinel)
    metadata_list.append({"none_option": True})
    return (options, metadata_list)


def truncate_question(question: str) -> str:
    """Enforce Telegram's 300-character limit on poll questions."""
    cleaned = " ".join(str(question or "").split()) or "Select your preferences:"
    return _truncate_text(cleaned, 300)


def build_profile_context(db: Database) -> ProfileContext:
    """Assemble global/legacy preference profile, weights, and feedback titles."""
    summary = db.get_state("preference_summary") or DEFAULT_PREFERENCE_SUMMARY
    rows = db.get_all_topic_weights()
    top_topics = [(row["topic"], float(row["weight"])) for row in rows[:8]]
    low_topics = [
        (row["topic"], float(row["weight"]))
        for row in reversed(rows)
        if float(row["weight"]) < 0.9
    ][:5]

    core_names = [row["topic"] for row in rows if row["category"] == "core"]
    learned_names = [row["topic"] for row in rows if row["category"] != "core"][:25]
    known_topics = core_names + [t for t in learned_names if t not in core_names]

    liked_titles, skipped_titles = db.get_recent_titles_for_profile(limit=5)
    return ProfileContext(
        preference_summary=summary,
        top_topics=top_topics,
        low_topics=low_topics,
        known_topics=known_topics,
        liked_titles=liked_titles,
        skipped_titles=skipped_titles,
    )


def build_user_profile_context(db: Database, user_id: int) -> ProfileContext:
    """Assemble personalized preference profile, weights, and feedback titles for user_id."""
    summary = db.get_user_preference_summary(user_id) or DEFAULT_PREFERENCE_SUMMARY
    rows = db.get_user_topic_weights(user_id)
    if not rows:
        return build_profile_context(db)

    top_topics = [(row["topic"], float(row["weight"])) for row in rows[:8]]
    low_topics = [
        (row["topic"], float(row["weight"]))
        for row in reversed(rows)
        if float(row["weight"]) < 0.9
    ][:5]

    core_names = [row["topic"] for row in rows if row["category"] == "core"]
    learned_names = [row["topic"] for row in rows if row["category"] != "core"][:25]
    known_topics = core_names + [t for t in learned_names if t not in core_names]

    liked_titles, skipped_titles = db.get_user_recent_titles_for_profile(user_id, limit=5)
    return ProfileContext(
        preference_summary=summary,
        top_topics=top_topics,
        low_topics=low_topics,
        known_topics=known_topics,
        liked_titles=liked_titles,
        skipped_titles=skipped_titles,
    )
