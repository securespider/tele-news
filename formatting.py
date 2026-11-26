"""Telegram HTML formatting, message chunking, and signed inline action keyboards."""

from __future__ import annotations

import html
import json
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from llm import LiteratureReviewReport, PaperComparisonReport, PaperReview, Stage2Summary
from security import CallbackSigner

MAX_ARTICLE_HTML_CHARS = 3500
MAX_TELEGRAM_CHUNK_CHARS = 3800


def format_weight_bar(weight: float, width: int = 8) -> str:
    """Render an ASCII bar representing a topic weight in [0.1, 3.0]."""
    ratio = max(0.0, min(1.0, (float(weight) - 0.1) / 2.9))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _safe_href(url: str | None) -> str:
    cleaned = (url or "").strip()
    if cleaned.lower().startswith(("http://", "https://")):
        return html.escape(cleaned, quote=True)
    return "https://news.ycombinator.com"


def _join_html_lines(lines: list[str], max_len: int = MAX_TELEGRAM_CHUNK_CHARS) -> str:
    """Join complete HTML lines up to max_len without slicing mid-tag."""
    flat_lines = [sub for item in lines for sub in item.split("\n")]
    suffix = "\n<i>... (truncated)</i>"
    kept: list[str] = []
    current_len = 0
    for idx, line in enumerate(flat_lines):
        added_len = len(line) + (1 if kept else 0)
        remaining = len(flat_lines) - idx - 1
        reserve = len(suffix) if remaining > 0 else 0
        if current_len + added_len + reserve <= max_len:
            kept.append(line)
            current_len += added_len
        else:
            if not kept:
                return line[:max_len]
            return "\n".join(kept) + suffix
    return "\n".join(kept)


def chunk_html_blocks(
    blocks: list[str], *, header_html: str = "", max_len: int = MAX_TELEGRAM_CHUNK_CHARS
) -> list[str]:
    """Pack HTML blocks into Telegram-safe message chunks under max_len."""
    if not blocks:
        return [header_html] if header_html else []

    chunks: list[str] = []
    current = header_html.strip()

    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(block) <= max_len:
                current = block
            else:
                chunks.append(_join_html_lines([block], max_len=max_len))
                current = ""

    if current:
        chunks.append(current)
    return chunks


def chunk_digest_messages(
    header_html: str, article_blocks: list[str], max_len: int = MAX_TELEGRAM_CHUNK_CHARS
) -> list[str]:
    """Pack digest header and article HTML blocks into Telegram-safe message chunks."""
    return chunk_html_blocks(article_blocks, header_html=header_html, max_len=max_len)


def _decode_authors(article: dict[str, Any]) -> list[str]:
    raw = article.get("authors_json")
    if not raw:
        return []
    try:
        loaded = json.loads(str(raw))
        if isinstance(loaded, list):
            return [str(a).strip() for a in loaded if str(a).strip()]
    except (TypeError, ValueError):
        pass
    return []


def format_article_html(
    index: int,
    article: dict[str, Any],
    summary: Stage2Summary,
) -> str:
    """Render a single digest entry (paper or news) as Telegram HTML <= 3500 chars."""
    title_raw = " ".join(str(article.get("title") or "Untitled").split())[:220]
    escaped_title = html.escape(title_raw, quote=True)
    escaped_url = _safe_href(article.get("url"))
    score = float(article.get("relevance_score") or 0.0)
    topics = summary.topics or []
    escaped_topics = html.escape(", ".join(topics[:4]) or "AI", quote=True)

    is_paper = article.get("content_type") == "paper" or bool(article.get("arxiv_id"))
    badge = "📄" if is_paper else "📰"

    meta_bits: list[str] = [f"Score: {score:.2f}", f"Topics: {escaped_topics}"]
    if article.get("arxiv_id"):
        meta_bits.append(f"arXiv:{html.escape(str(article['arxiv_id']))}")
    if article.get("hf_upvotes") is not None:
        meta_bits.append(f"HF▲{int(article['hf_upvotes'])}")
    if article.get("citation_count") is not None:
        meta_bits.append(f"Cites:{int(article['citation_count'])}")

    authors = _decode_authors(article)
    authors_line = ""
    if authors:
        short_authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        authors_line = f"<i>Authors: {html.escape(short_authors)}</i>"
        if article.get("venue"):
            authors_line += f" · <i>{html.escape(str(article['venue'])[:60])}</i>"

    bullets = summary.summary_bullets[:3] or ["Summary unavailable."]
    why = " ".join((summary.why_relevant or "").split())[:350]
    pulse = " ".join((summary.hn_pulse or "").split())[:350] if summary.hn_pulse else None
    comments_url = article.get("comments_url")
    code_url = article.get("code_url")
    short_id = str(article.get("id") or "")[:10]

    def _assemble(bullet_limit: int) -> str:
        lines = [
            f'<b>{index}. {badge} <a href="{escaped_url}">{escaped_title}</a></b>',
            f"<i>{' | '.join(meta_bits)}</i>",
        ]
        if authors_line:
            lines.append(authors_line)
        for b in bullets:
            clean_b = " ".join(b.split())
            if len(clean_b) > bullet_limit:
                clean_b = clean_b[: max(10, bullet_limit - 3)].rstrip() + "..."
            lines.append(f"• {html.escape(clean_b, quote=True)}")
        if why:
            lines.append(f"<i>Why relevant:</i> {html.escape(why, quote=True)}")
        if pulse:
            pulse_line = f"<i>HN Pulse:</i> {html.escape(pulse, quote=True)}"
            if comments_url:
                pulse_line += f' (<a href="{_safe_href(comments_url)}">HN Discussion</a>)'
            lines.append(pulse_line)
        footer_parts = []
        if code_url:
            footer_parts.append(f'<a href="{_safe_href(code_url)}">Code</a>')
        if is_paper and (article.get("arxiv_id") or short_id):
            ref = article.get("arxiv_id") or short_id
            footer_parts.append(f"<code>/review {html.escape(str(ref))}</code>")
        elif short_id:
            footer_parts.append(f"<code>/save {html.escape(short_id)}</code>")
        if footer_parts:
            lines.append(" · ".join(footer_parts))
        return "\n".join(lines)

    for limit in (500, 300, 180, 90):
        rendered = _assemble(limit)
        if len(rendered) <= MAX_ARTICLE_HTML_CHARS:
            return rendered

    minimal = (
        f'<b>{index}. {badge} <a href="{escaped_url}">{escaped_title[:120]}</a></b>\n'
        f"<i>Score: {score:.2f} | Topics: {escaped_topics[:100]}</i>"
    )
    return minimal[:MAX_ARTICLE_HTML_CHARS]


def build_article_action_keyboard(
    signer: CallbackSigner,
    user_id: int,
    article_id: str,
    *,
    is_paper: bool = False,
    in_library: bool = False,
) -> InlineKeyboardMarkup:
    """Build HMAC-signed inline action buttons bound to user_id."""
    ref = article_id[:32]
    row: list[InlineKeyboardButton] = []
    if in_library:
        row.append(
            InlineKeyboardButton(
                "✅ Saved (Remove)",
                callback_data=signer.sign(user_id, "rm", ref),
            )
        )
    else:
        row.append(
            InlineKeyboardButton(
                "🔖 Save",
                callback_data=signer.sign(user_id, "sv", ref),
            )
        )
    if is_paper:
        row.append(
            InlineKeyboardButton(
                "🔬 Review",
                callback_data=signer.sign(user_id, "rv", ref),
            )
        )
    row.append(
        InlineKeyboardButton(
            "📋 BibTeX",
            callback_data=signer.sign(user_id, "bx", ref),
        )
    )
    return InlineKeyboardMarkup([row])


def build_digest_quick_keyboard(
    signer: CallbackSigner,
    user_id: int,
    articles: list[dict[str, Any]],
) -> InlineKeyboardMarkup | None:
    """Build compact multi-row signed Save/Review keyboard for digest items."""
    if not articles:
        return None
    rows: list[list[InlineKeyboardButton]] = []
    for idx, art in enumerate(articles[:5], start=1):
        art_id = str(art.get("id") or "")[:32]
        if not art_id:
            continue
        is_paper = art.get("content_type") == "paper" or bool(art.get("arxiv_id"))
        btn_row = [
            InlineKeyboardButton(
                f"🔖 Save #{idx}",
                callback_data=signer.sign(user_id, "sv", art_id),
            )
        ]
        if is_paper:
            btn_row.append(
                InlineKeyboardButton(
                    f"🔬 Review #{idx}",
                    callback_data=signer.sign(user_id, "rv", art_id),
                )
            )
        btn_row.append(
            InlineKeyboardButton(
                f"📋 BibTeX #{idx}",
                callback_data=signer.sign(user_id, "bx", art_id),
            )
        )
        rows.append(btn_row)
    return InlineKeyboardMarkup(rows) if rows else None


def format_paper_review_html(paper: dict[str, Any], review: PaperReview) -> list[str]:
    """Render a structured PaperReview as Telegram HTML chunks."""
    title = html.escape(" ".join(str(paper.get("title") or "Untitled").split())[:220], quote=True)
    url = _safe_href(paper.get("url"))
    meta_parts: list[str] = [f"Maturity: {html.escape(review.maturity_level)}"]
    if paper.get("arxiv_id"):
        meta_parts.append(f"arXiv:{html.escape(str(paper['arxiv_id']))}")
    if paper.get("citation_count") is not None:
        meta_parts.append(f"Cites: {int(paper['citation_count'])}")
    if paper.get("hf_upvotes") is not None:
        meta_parts.append(f"HF▲{int(paper['hf_upvotes'])}")

    blocks: list[str] = [
        f'<b>🔬 Paper Review: <a href="{url}">{title}</a></b>\n<i>{" | ".join(meta_parts)}</i>',
        f"<b>TL;DR:</b> {html.escape(review.tldr)}",
        f"<b>Problem:</b> {html.escape(review.problem)}",
        f"<b>Methodology:</b> {html.escape(review.methodology)}",
    ]

    if review.key_results:
        res_lines = ["<b>Key Results:</b>"] + [
            f"• {html.escape(r)}" for r in review.key_results
        ]
        blocks.append("\n".join(res_lines))

    if review.limitations:
        lim_lines = ["<b>Limitations &amp; Risks:</b>"] + [
            f"• {html.escape(lim)}" for lim in review.limitations
        ]
        blocks.append("\n".join(lim_lines))

    blocks.append(f"<b>Practical Takeaways:</b> {html.escape(review.practical_takeaways)}")
    blocks.append(f"<b>Reproducibility:</b> {html.escape(review.reproducibility_notes)}")

    if paper.get("code_url"):
        blocks.append(f'<b>Code:</b> <a href="{_safe_href(paper["code_url"])}">Repository Link</a>')
    if review.topics:
        blocks.append(f"<i>Topics: {html.escape(', '.join(review.topics))}</i>")

    return chunk_html_blocks(blocks)


def format_litreview_html(
    query: str,
    report: LiteratureReviewReport,
    papers: list[dict[str, Any]],
) -> list[str]:
    """Render a LiteratureReviewReport with grounded clickable paper references."""
    header = (
        f"<b>📚 Literature Review: {html.escape(query[:150])}</b>\n"
        f"<i>Grounded in {len(papers)} papers</i>"
    )
    blocks: list[str] = [
        f"<b>Executive Summary</b>\n{html.escape(report.executive_summary)}",
        f"<b>Consensus &amp; Debates</b>\n{html.escape(report.consensus_and_debates)}",
    ]

    if report.comparison_rows:
        comp_lines = ["<b>Comparison Breakdown</b>"]
        for row in report.comparison_rows[:8]:
            comp_lines.append(
                f"• <b>{html.escape(row.paper_ref)}</b> — "
                f"<i>Method:</i> {html.escape(row.method)} | "
                f"<i>Data:</i> {html.escape(row.benchmark_or_data)} | "
                f"<i>Finding:</i> {html.escape(row.key_finding)} | "
                f"<i>Limit:</i> {html.escape(row.limitation)}"
            )
        blocks.append("\n".join(comp_lines))

    if report.research_gaps:
        blocks.append(
            "<b>Open Research Gaps</b>\n"
            + "\n".join(f"• {html.escape(g)}" for g in report.research_gaps)
        )

    if report.recommended_reading_order:
        blocks.append(
            "<b>Recommended Reading Order</b>\n"
            + "\n".join(
                f"{idx}. {html.escape(step)}"
                for idx, step in enumerate(report.recommended_reading_order, start=1)
            )
        )

    ref_lines = ["<b>Grounded References</b>"]
    for idx, p in enumerate(papers, start=1):
        title = html.escape(str(p.get("title") or "Untitled")[:180], quote=True)
        url = _safe_href(p.get("url"))
        extra = f" (arXiv:{html.escape(str(p['arxiv_id']))})" if p.get("arxiv_id") else ""
        ref_lines.append(f'<b>[P{idx}]</b> <a href="{url}">{title}</a>{extra}')
    blocks.append("\n".join(ref_lines))

    return chunk_html_blocks(blocks, header_html=header)


def format_paper_comparison_html(
    papers: list[dict[str, Any]], report: PaperComparisonReport
) -> list[str]:
    """Render a side-by-side PaperComparisonReport."""
    header_lines = ["<b>⚖️ Side-by-Side Paper Comparison</b>"]
    for idx, p in enumerate(papers, start=1):
        title = html.escape(str(p.get("title") or "Untitled")[:140], quote=True)
        url = _safe_href(p.get("url"))
        header_lines.append(f'<b>[P{idx}]</b> <a href="{url}">{title}</a>')

    blocks: list[str] = [
        "\n".join(header_lines),
        f"<b>Overview</b>\n{html.escape(report.summary)}",
    ]

    for dim in report.dimensions:
        dim_lines = [f"<b>{html.escape(dim.dimension)}</b>"]
        for idx, val in enumerate(dim.assessments, start=1):
            dim_lines.append(f"• <b>[P{idx}]</b> {html.escape(val)}")
        blocks.append("\n".join(dim_lines))

    blocks.append(f"<b>Practitioner Verdict</b>\n{html.escape(report.verdict_for_practitioners)}")
    return chunk_html_blocks(blocks)


def format_library_html(
    items: list[dict[str, Any]], page_title: str = "📚 Your Research Library"
) -> list[str]:
    """Render saved library items as Telegram HTML chunks."""
    if not items:
        return [
            f"<b>{html.escape(page_title)}</b>\n\n"
            "<i>No saved papers or articles yet. Use /save &lt;arxiv_id|url|id&gt; or tap 🔖 Save on any digest item!</i>"
        ]

    blocks: list[str] = []
    for item in items:
        lib_id = item["id"]
        status = item.get("status") or "to_read"
        badge = {"to_read": "📌", "reading": "📖", "read": "✅", "archived": "🗄️"}.get(status, "📌")
        title = html.escape(str(item.get("title") or "Untitled")[:160], quote=True)
        url = _safe_href(item.get("url"))
        tags: list[str] = []
        try:
            tags = json.loads(item.get("tags_json") or "[]")
        except (TypeError, ValueError):
            pass
        tag_str = f" | tags: {html.escape(', '.join(tags))}" if tags else ""
        arxiv_str = f" | arXiv:{html.escape(str(item['arxiv_id']))}" if item.get("arxiv_id") else ""
        note_line = (
            f"\n   📝 <i>{html.escape(str(item['notes'])[:180])}</i>"
            if (item.get("notes") or "").strip()
            else ""
        )
        blocks.append(
            f'<b>#{lib_id}</b> {badge} <a href="{url}">{title}</a>\n'
            f"   <code>[{html.escape(status)}]</code>{arxiv_str}{tag_str}{note_line}"
        )

    footer = (
        "<i>Commands: /read &lt;id&gt;, /note &lt;id&gt; &lt;text&gt;, "
        "/review &lt;id&gt;, /bibtex &lt;id&gt;, /export [bibtex|markdown]</i>"
    )
    blocks.append(footer)
    return chunk_html_blocks(blocks, header_html=f"<b>{html.escape(page_title)} ({len(items)})</b>")


def format_trending_html(
    hf_papers: list[dict[str, Any]], org_trending: list[dict[str, Any]]
) -> str:
    """Format Hugging Face Daily Papers and anonymized Org Trending picks."""
    lines = ["<b>🔥 Trending in AI Research</b>"]

    lines.append("\n<b>🤗 Hugging Face Daily Papers</b>")
    if hf_papers:
        for idx, p in enumerate(hf_papers[:6], start=1):
            title = html.escape(str(p.get("title") or "Untitled")[:140], quote=True)
            url = _safe_href(p.get("url"))
            up = f"▲{int(p['hf_upvotes'])}" if p.get("hf_upvotes") is not None else ""
            arxiv = p.get("arxiv_id") or ""
            review_cmd = f" · <code>/review {html.escape(str(arxiv))}</code>" if arxiv else ""
            lines.append(f'{idx}. <a href="{url}">{title}</a> <i>{up}</i>{review_cmd}')
    else:
        lines.append("<i>No recent HF Daily Papers cached yet.</i>")

    lines.append("\n<b>🏢 Org-Wide Top Picks (7d)</b>")
    if org_trending:
        for idx, item in enumerate(org_trending[:5], start=1):
            title = html.escape(str(item.get("title") or "Untitled")[:140], quote=True)
            url = _safe_href(item.get("url"))
            likes = int(item.get("org_likes") or 0)
            saves = int(item.get("org_saves") or 0)
            lines.append(f'{idx}. <a href="{url}">{title}</a> <i>(👍 {likes} · 🔖 {saves})</i>')
    else:
        lines.append("<i>No team poll votes or library saves recorded in the past 7 days yet.</i>")

    return _join_html_lines(lines)
