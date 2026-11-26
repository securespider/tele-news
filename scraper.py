"""Two-stage RSS scraper, URL normalizer, hardened SSRF fetcher, trafilatura, and HN Algolia."""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
import trafilatura

from security import (
    PayloadTooLargeError,
    SSRFBlockedError,
    create_safe_async_client,
    is_safe_public_url,
    safe_fetch_bytes,
    safe_fetch_text,
)

logger = logging.getLogger(__name__)

MAX_RSS_BYTES = 5 * 1024 * 1024
MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_FULL_TEXT_CHARS = 12_000
MAX_RSS_SNIPPET_CHARS = 800
MAX_HN_COMMENTS_TOTAL_CHARS = 3_000
MAX_HN_COMMENT_CHARS = 600

RSS_USER_AGENT = "DailyAI-RSS-Agent/2.0 (+https://github.com)"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_TRACKING_EXACT_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}
_HN_ITEM_RE = re.compile(r"news\.ycombinator\.com/item\?id=(\d+)")
_HN_URL_RE = re.compile(r"https?://news\.ycombinator\.com/item\?id=\d+")
_ARXIV_PDF_RE = re.compile(
    r"^https?://(?:www\.)?arxiv\.org/pdf/([^\s/?#]+?)(?:\.pdf)?$", re.IGNORECASE
)
_ARXIV_ID_IN_URL_RE = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_GITHUB_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
    re.IGNORECASE,
)
_BLOCK_TAGS_RE = re.compile(
    r"</?(?:p|br|li|div|blockquote|pre|tr|h[1-6])\b[^>]*>", re.IGNORECASE
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


class InvalidFeedError(ValueError):
    """Raised when a fetched response is not a valid RSS or Atom feed."""


def normalize_url(url: str) -> str:
    """Canonicalize URL by lowercasing host, stripping tracking params, fragments, and trailing slash."""
    stripped = (url or "").strip()
    if not stripped:
        return ""
    parts = urlsplit(stripped)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    kept_params = []
    for key, val in parse_qsl(parts.query, keep_blank_values=True):
        lowered_key = key.lower()
        if lowered_key.startswith("utm_") or lowered_key in _TRACKING_EXACT_PARAMS:
            continue
        kept_params.append((key, val))
    kept_params.sort(key=lambda item: (item[0], item[1]))
    query = urlencode(kept_params, doseq=True)

    path = parts.path
    if path and path != "/":
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, query, ""))


def compute_article_id(url: str, guid: str | None = None) -> str:
    """Return deterministic 32-char SHA-256 prefix from normalized URL or fallback GUID."""
    canonical = normalize_url(url) or normalize_url(guid or "") or (guid or "").strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def rewrite_fetch_url(url: str) -> str:
    """Rewrite arXiv PDF links to their HTML abstract page for trafilatura extraction."""
    cleaned = (url or "").strip()
    parts = urlsplit(cleaned)
    base_no_query = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    match = _ARXIV_PDF_RE.match(base_no_query)
    if match:
        paper_id = match.group(1)
        return f"https://arxiv.org/abs/{paper_id}"
    return cleaned


def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags safely before unescaping entities and collapsing whitespace."""
    if not raw_html:
        return ""
    spaced = _BLOCK_TAGS_RE.sub(" ", raw_html)
    stripped = _ANY_TAG_RE.sub("", spaced)
    unescaped = html.unescape(stripped)
    return " ".join(unescaped.split())


def extract_github_urls(text: str | None) -> list[str]:
    """Extract unique GitHub/GitLab repository URLs from text or HTML snippets."""
    if not text:
        return []
    found: list[str] = []
    for match in _GITHUB_URL_RE.findall(text):
        cleaned = match.rstrip(".,;:)\"'")
        if cleaned not in found:
            found.append(cleaned)
    return found


def extract_hn_item_id(comments_url: str | None, url: str | None = None) -> str | None:
    """Extract numeric Hacker News item ID from comments_url first, then url."""
    for candidate in (comments_url, url):
        if not candidate:
            continue
        match = _HN_ITEM_RE.search(candidate)
        if match:
            return match.group(1)
    return None


def _extract_comments_url(entry: Any, url: str) -> str | None:
    comments = (entry.get("comments") or "").strip()
    if comments:
        return comments

    guid = (entry.get("id") or "").strip()
    if guid and _HN_ITEM_RE.search(guid):
        return guid

    if _HN_ITEM_RE.search(url):
        return url

    summary_raw = entry.get("summary") or entry.get("description") or ""
    match = _HN_URL_RE.search(summary_raw)
    if match:
        return match.group(0)
    return None


def _extract_entry_authors(entry: Any) -> list[str]:
    authors: list[str] = []
    raw_authors = entry.get("authors")
    if isinstance(raw_authors, list):
        for item in raw_authors:
            name = str((item or {}).get("name") or "").strip()
            if name and name not in authors:
                authors.append(name)
    if not authors and entry.get("author"):
        for part in str(entry["author"]).split(","):
            name = clean_html_text(part).strip()
            if name and name not in authors:
                authors.append(name)
    return authors[:20]


def is_valid_parsed_feed(parsed: Any) -> bool:
    """Return True if feedparser recognized an RSS/Atom version or extracted entries."""
    version = str(getattr(parsed, "version", "") or "").strip()
    entries = getattr(parsed, "entries", None) or []
    return bool(version) or len(entries) > 0


def parse_feed_entries(
    xml_text: str,
    source_feed: str,
    *,
    max_age_hours: int = 72,
    max_entries_per_feed: int = 30,
    now: datetime | None = None,
    strict: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse RSS/Atom XML into normalized article/paper dicts using UTC calendar.timegm."""
    parsed = feedparser.parse(xml_text)
    if strict and not is_valid_parsed_feed(parsed):
        raise InvalidFeedError("Response is not a valid RSS or Atom feed")
    feed_title_raw = parsed.feed.get("title") if getattr(parsed, "feed", None) else None
    feed_title = " ".join(str(feed_title_raw or "").split()) or None

    ref_now = now or datetime.now(timezone.utc)
    cutoff = ref_now - timedelta(hours=max_age_hours)

    items: list[dict[str, Any]] = []
    for entry in getattr(parsed, "entries", []):
        title = " ".join(str(entry.get("title") or "").split())
        url = str(entry.get("link") or "").strip()
        if not title or not url:
            continue

        time_struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if time_struct is not None:
            ts = calendar.timegm(time_struct)
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if pub_dt < cutoff:
                continue
            published_at = pub_dt.isoformat()
        else:
            published_at = ref_now.isoformat()

        comments_url = _extract_comments_url(entry, url)
        raw_summary = entry.get("summary") or entry.get("description") or ""
        rss_snippet = clean_html_text(str(raw_summary))[:MAX_RSS_SNIPPET_CHARS]
        article_id = compute_article_id(url, entry.get("id"))

        arxiv_match = _ARXIV_ID_IN_URL_RE.search(url)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None
        content_type = "paper" if arxiv_id else "news"
        authors = _extract_entry_authors(entry)
        code_urls = extract_github_urls(f"{raw_summary} {url}")

        items.append(
            {
                "id": article_id,
                "url": url,
                "comments_url": comments_url,
                "title": title,
                "source_feed": source_feed,
                "published_at": published_at,
                "rss_snippet": rss_snippet,
                "content_type": content_type,
                "arxiv_id": arxiv_id,
                "authors_json": json.dumps(authors) if authors else None,
                "code_url": code_urls[0] if code_urls else None,
            }
        )

    items.sort(key=lambda item: item["published_at"], reverse=True)
    return feed_title, items[:max_entries_per_feed]


async def fetch_new_rss_entries(
    feeds: list[dict[str, Any]],
    *,
    max_age_hours: int = 72,
    max_entries_per_feed: int = 30,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fetch all enabled RSS/Atom feeds concurrently with ETag/Last-Modified and SSRF defense."""
    if not feeds:
        return ([], {})

    semaphore = asyncio.Semaphore(5)
    all_entries: list[dict[str, Any]] = []
    feed_updates: dict[str, dict[str, Any]] = {}

    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )

    async def _fetch_one(feed: dict[str, Any]) -> None:
        feed_url = feed["url"]
        if not is_safe_public_url(feed_url):
            feed_updates[feed_url] = {"ok": False, "error": "Rejected unsafe feed URL"}
            return

        req_headers: dict[str, str] = {}
        if feed.get("etag"):
            req_headers["If-None-Match"] = feed["etag"]
        if feed.get("last_modified"):
            req_headers["If-Modified-Since"] = feed["last_modified"]

        async with semaphore:
            try:
                status_code, _, resp_headers, xml_text = await safe_fetch_text(
                    active_client,
                    feed_url,
                    headers=req_headers,
                    max_bytes=MAX_RSS_BYTES,
                )
                if status_code == 304:
                    feed_updates[feed_url] = {
                        "ok": True,
                        "etag": resp_headers.get("ETag") or feed.get("etag"),
                        "last_modified": resp_headers.get("Last-Modified")
                        or feed.get("last_modified"),
                        "title": None,
                    }
                    return
                etag = resp_headers.get("ETag")
                last_modified = resp_headers.get("Last-Modified")
            except Exception as exc:
                logger.warning("Failed fetching RSS feed %s: %s", feed_url, exc)
                feed_updates[feed_url] = {"ok": False, "error": str(exc)}
                return

        try:
            feed_title, parsed_entries = parse_feed_entries(
                xml_text,
                feed_url,
                max_age_hours=max_age_hours,
                max_entries_per_feed=max_entries_per_feed,
                strict=True,
            )
        except InvalidFeedError as exc:
            logger.warning("Invalid RSS/Atom feed %s: %s", feed_url, exc)
            feed_updates[feed_url] = {"ok": False, "error": str(exc)}
            return
        all_entries.extend(parsed_entries)
        feed_updates[feed_url] = {
            "ok": True,
            "etag": etag,
            "last_modified": last_modified,
            "title": feed_title,
        }

    try:
        await asyncio.gather(*(_fetch_one(feed) for feed in feeds))
    finally:
        if owns_client:
            await active_client.aclose()

    return (all_entries, feed_updates)


async def validate_rss_feed_url(
    url: str, *, client: httpx.AsyncClient | None = None
) -> tuple[bool, str | None, str | None]:
    """Fetch and verify that a public URL serves a valid RSS or Atom feed."""
    if not is_safe_public_url(url):
        return (
            False,
            None,
            "Invalid or unsafe URL. Please supply a public http:// or https:// feed URL.",
        )

    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )

    try:
        _, _, _, xml_text = await safe_fetch_text(
            active_client,
            url,
            max_bytes=MAX_RSS_BYTES,
        )
    except Exception as exc:
        return (False, None, f"Could not fetch feed: {exc}")
    finally:
        if owns_client:
            await active_client.aclose()

    parsed = feedparser.parse(xml_text)
    if not is_valid_parsed_feed(parsed):
        return (False, None, "URL did not return a valid RSS or Atom feed.")

    feed_title_raw = parsed.feed.get("title") if getattr(parsed, "feed", None) else None
    feed_title = " ".join(str(feed_title_raw or "").split()) or None
    return (True, feed_title, None)


async def fetch_full_text(url: str, client: httpx.AsyncClient | None = None) -> str | None:
    """Download HTML (with SSRF, DNS-rebinding & 3MB caps) and extract clean text via trafilatura."""
    target_url = rewrite_fetch_url(url)
    if not is_safe_public_url(target_url):
        logger.warning("Skipping unsafe article URL: %s", target_url)
        return None

    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": BROWSER_USER_AGENT},
    )

    try:
        _, final_url, resp_headers, html_bytes = await safe_fetch_bytes(
            active_client,
            target_url,
            max_bytes=MAX_HTML_BYTES,
        )
        content_type = (resp_headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None
        html_body = html_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed fetching article text for %s: %s", target_url, exc)
        return None
    finally:
        if owns_client:
            await active_client.aclose()

    try:
        extracted = await asyncio.to_thread(
            trafilatura.extract,
            html_body,
            url=final_url,
            include_comments=False,
            include_tables=True,
        )
    except Exception as exc:
        logger.warning("trafilatura failed for %s: %s", target_url, exc)
        return None

    if not extracted:
        return None
    cleaned = extracted.strip()
    return cleaned[:MAX_FULL_TEXT_CHARS] if cleaned else None


async def fetch_hn_item(
    hn_item_id: str, client: httpx.AsyncClient | None = None
) -> tuple[str | None, str | None]:
    """Fetch Hacker News self-post text and up to 5 clean root comments via Algolia API."""
    if not str(hn_item_id).isdigit():
        return (None, None)
    api_url = f"https://hn.algolia.com/api/v1/items/{hn_item_id}"
    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=10.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )

    try:
        _, _, _, raw_bytes = await safe_fetch_bytes(
            active_client,
            api_url,
            max_bytes=MAX_HTML_BYTES,
        )
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("Failed fetching HN Algolia item %s: %s", hn_item_id, exc)
        return (None, None)
    finally:
        if owns_client:
            await active_client.aclose()

    raw_post_text = data.get("text")
    self_post_text = (
        clean_html_text(str(raw_post_text))[:MAX_FULL_TEXT_CHARS] if raw_post_text else None
    )
    if self_post_text == "":
        self_post_text = None

    comment_lines: list[str] = []
    total_chars = 0
    for child in data.get("children") or []:
        author = child.get("author")
        raw_text = child.get("text")
        if not author or not raw_text:
            continue
        cleaned = clean_html_text(str(raw_text))
        if not cleaned:
            continue
        if len(cleaned) > MAX_HN_COMMENT_CHARS:
            cleaned = cleaned[: MAX_HN_COMMENT_CHARS - 3].rstrip() + "..."
        line = f"- @{author}: {cleaned}"
        if total_chars + len(line) > MAX_HN_COMMENTS_TOTAL_CHARS and comment_lines:
            break
        comment_lines.append(line[:MAX_HN_COMMENTS_TOTAL_CHARS])
        total_chars += len(line) + 1
        if len(comment_lines) >= 5:
            break

    comments_block = "\n".join(comment_lines) if comment_lines else None
    return (self_post_text, comments_block)


async def fetch_hn_comments(
    comments_url: str | None,
    url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Extract up to 5 top Hacker News comments if a discussion item ID exists."""
    hn_item_id = extract_hn_item_id(comments_url, url)
    if not hn_item_id:
        return None
    _, comments_block = await fetch_hn_item(hn_item_id, client=client)
    return comments_block


async def enrich_article_content(
    article: dict[str, Any], client: httpx.AsyncClient | None = None
) -> tuple[str | None, str | None]:
    """Fetch full article text and Hacker News comments for one digest candidate."""
    url = article.get("url") or ""
    comments_url = article.get("comments_url")
    url_hn_id = extract_hn_item_id(None, url)

    if url_hn_id:
        self_text, comments = await fetch_hn_item(url_hn_id, client=client)
        if self_text:
            return (self_text, comments)
        full_text = await fetch_full_text(url, client=client)
        return (full_text, comments)

    comments_hn_id = extract_hn_item_id(comments_url, None)
    if comments_hn_id:
        full_text, (_, comments) = await asyncio.gather(
            fetch_full_text(url, client=client),
            fetch_hn_item(comments_hn_id, client=client),
        )
        return (full_text, comments)

    full_text = await fetch_full_text(url, client=client)
    return (full_text, None)


async def enrich_articles_batch(
    articles: list[dict[str, Any]], max_concurrency: int = 3
) -> dict[str, tuple[str | None, str | None]]:
    """Enrich digest articles in bounded parallel using a shared hardened httpx client."""
    if not articles:
        return {}

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    results: dict[str, tuple[str | None, str | None]] = {}

    async with create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": BROWSER_USER_AGENT},
    ) as client:

        async def _run_one(article: dict[str, Any]) -> None:
            async with semaphore:
                results[article["id"]] = await enrich_article_content(article, client=client)

        await asyncio.gather(*(_run_one(article) for article in articles))

    return results
