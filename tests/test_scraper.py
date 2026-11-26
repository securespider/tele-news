"""Tests for URL normalization, SSRF guard, RSS/HN parsing, and Stage 2 enrichment."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

import scraper
from scraper import (
    clean_html_text,
    compute_article_id,
    enrich_article_content,
    fetch_full_text,
    fetch_hn_comments,
    fetch_new_rss_entries,
    is_safe_public_url,
    normalize_url,
    parse_feed_entries,
    rewrite_fetch_url,
    validate_rss_feed_url,
)


def test_normalize_url_and_deterministic_id() -> None:
    u1 = "HTTPS://Example.COM/ai/post/?utm_source=hn&utm_medium=rss&fbclid=abc&p=42#comments"
    u2 = "https://example.com/ai/post?p=42"
    assert normalize_url(u1) == "https://example.com/ai/post?p=42"
    assert compute_article_id(u1) == compute_article_id(u2)

    # Must keep Hacker News item id and root slash
    hn = "https://news.ycombinator.com/item?id=12345&utm_campaign=daily"
    assert normalize_url(hn) == "https://news.ycombinator.com/item?id=12345"
    assert normalize_url("https://example.com/") == "https://example.com/"


def test_is_safe_public_url_and_arxiv_rewrite() -> None:
    assert is_safe_public_url("https://hnrss.org/frontpage") is True
    assert is_safe_public_url("http://127.0.0.1:8080/feed") is False
    assert is_safe_public_url("http://localhost:11434/v1") is False
    assert is_safe_public_url("http://192.168.1.50/rss") is False
    assert is_safe_public_url("http://169.254.169.254/latest/meta-data/") is False
    assert is_safe_public_url("file:///etc/passwd") is False
    assert is_safe_public_url("https://myserver.local/rss") is False

    assert (
        rewrite_fetch_url("https://arxiv.org/pdf/2501.12345.pdf")
        == "https://arxiv.org/abs/2501.12345"
    )
    assert (
        rewrite_fetch_url("http://www.arxiv.org/pdf/2501.12345v2?download=1")
        == "https://arxiv.org/abs/2501.12345v2"
    )


def test_clean_html_text_preserves_escaped_code_brackets() -> None:
    raw = "<p>First line<br/>Use &lt;tool_call&gt; &amp; JSON</p>"
    assert clean_html_text(raw) == "First line Use <tool_call> & JSON"


def test_parse_hnrss_xml_utc_and_age_filtering() -> None:
    now = datetime(2026, 11, 22, 12, 0, 0, tzinfo=timezone.utc)
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Hacker News: Front Page</title>
        <item>
          <title>New Agentic Coding Benchmark</title>
          <link>https://example.org/agent-bench</link>
          <comments>https://news.ycombinator.com/item?id=445566</comments>
          <pubDate>Sun, 22 Nov 2026 08:00:00 GMT</pubDate>
          <description><![CDATA[<p>Article URL: <a href="https://example.org/agent-bench">link</a> &amp; benchmarks</p>]]></description>
        </item>
        <item>
          <title>Ancient Post</title>
          <link>https://example.org/old-post</link>
          <pubDate>Sun, 01 Nov 2026 08:00:00 GMT</pubDate>
          <description>Too old</description>
        </item>
      </channel>
    </rss>
    """
    feed_title, entries = parse_feed_entries(
        xml, "https://hnrss.org/frontpage", max_age_hours=72, max_entries_per_feed=30, now=now
    )
    assert feed_title == "Hacker News: Front Page"
    assert len(entries) == 1
    entry = entries[0]
    assert entry["url"] == "https://example.org/agent-bench"
    assert entry["comments_url"] == "https://news.ycombinator.com/item?id=445566"
    # Verify calendar.timegm keeps exact UTC hour (08:00+00:00, not local TZ shifted)
    assert entry["published_at"] == "2026-11-22T08:00:00+00:00"
    assert "benchmarks" in entry["rss_snippet"]
    assert "<p>" not in entry["rss_snippet"]


@pytest.mark.asyncio
async def test_fetch_new_rss_entries_304_and_error_isolation() -> None:
    rss_ok = """<?xml version="1.0"?><rss version="2.0"><channel><title>Good</title>
    <item><title>Fresh AI Paper</title><link>https://good.example/p1</link></item>
    </channel></rss>"""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://good.example/rss":
            return httpx.Response(200, text=rss_ok, headers={"ETag": '"v2"'})
        if str(request.url) == "https://cached.example/rss":
            assert request.headers.get("If-None-Match") == '"v1"'
            return httpx.Response(304, headers={"ETag": '"v1"'})
        return httpx.Response(500, text="Server error")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        feeds = [
            {"url": "https://good.example/rss", "etag": None, "last_modified": None},
            {"url": "https://cached.example/rss", "etag": '"v1"', "last_modified": None},
            {"url": "https://broken.example/rss", "etag": None, "last_modified": None},
        ]
        entries, updates = await fetch_new_rss_entries(feeds, client=client)

    assert len(entries) == 1
    assert entries[0]["title"] == "Fresh AI Paper"
    assert updates["https://good.example/rss"]["ok"] is True
    assert updates["https://good.example/rss"]["etag"] == '"v2"'
    assert updates["https://cached.example/rss"]["ok"] is True
    assert updates["https://broken.example/rss"]["ok"] is False


@pytest.mark.asyncio
async def test_fetch_full_text_extraction_arxiv_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_urls: list[str] = []

    def fake_extract(html_body: str, **kwargs: object) -> str | None:
        captured_urls.append(str(kwargs.get("url")))
        if "EMPTY" in html_body:
            return None
        return "Extracted technical body about sparse attention."

    monkeypatch.setattr(scraper.trafilatura, "extract", fake_extract)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://arxiv.org/abs/2501.00001":
            return httpx.Response(
                200,
                text="<html><body>Abstract</body></html>",
                headers={"Content-Type": "text/html; charset=utf-8"},
            )
        if url == "https://example.org/pdf":
            return httpx.Response(200, content=b"%PDF-1.4", headers={"Content-Type": "application/pdf"})
        return httpx.Response(404, text="Not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        text = await fetch_full_text("https://arxiv.org/pdf/2501.00001.pdf", client=client)
        assert text == "Extracted technical body about sparse attention."
        assert captured_urls == ["https://arxiv.org/abs/2501.00001"]

        assert await fetch_full_text("https://example.org/pdf", client=client) is None
        assert await fetch_full_text("https://example.org/missing", client=client) is None


@pytest.mark.asyncio
async def test_hn_algolia_comments_and_ask_hn_self_post() -> None:
    payload = {
        "id": 777,
        "text": "<p>Ask HN: Which local inference engine works best for <b>M4 Max</b>?</p>",
        "children": [
            {"author": None, "text": "[deleted]"},
            {"author": "alice", "text": "<p>llama.cpp &amp; MLX work great!</p>"},
            {"author": "bob", "text": "<p>vLLM for Linux boxes.</p>"},
            {"author": "carol", "text": ""},
            {"author": "dave", "text": "<p>Try &lt;q4_k_m&gt; quantization.</p>"},
            {"author": "erin", "text": "<p>Memory bandwidth is key.</p>"},
            {"author": "frank", "text": "<p>FlashAttention helps context length.</p>"},
            {"author": "grace", "text": "<p>Should be truncated past 5 comments.</p>"},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://hn.algolia.com/api/v1/items/777"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        comments = await fetch_hn_comments(
            "https://news.ycombinator.com/item?id=777", None, client=client
        )
        assert comments is not None
        lines = comments.splitlines()
        assert len(lines) == 5  # deleted/empty skipped, capped at 5
        assert lines[0] == "- @alice: llama.cpp & MLX work great!"
        assert lines[2] == "- @dave: Try <q4_k_m> quantization."

        # Ask HN self-post extracts full_text directly from Algolia text
        self_text, self_comments = await enrich_article_content(
            {"url": "https://news.ycombinator.com/item?id=777", "comments_url": None},
            client=client,
        )
        assert self_text == "Ask HN: Which local inference engine works best for M4 Max?"
        assert self_comments == comments


@pytest.mark.asyncio
async def test_validate_rss_feed_url_and_scrape_html_rejection() -> None:
    rss_xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>AI Weekly</title>
    <item><title>Post</title><link>https://example.org/p1</link></item></channel></rss>"""
    atom_xml = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
    <title>Research Atom</title></feed>"""
    empty_rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>Empty RSS</title></channel></rss>"""
    bozo_rss = """<?xml version="1.0"?><rss version="2.0"><channel><title>A & B</title>
    <item><title>Item</title><link>https://example.org/p2</link></item></channel></rss>"""
    html_page = "<!DOCTYPE html><html><head><title>Hacker News</title></head><body>Hi</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://good.example/rss":
            return httpx.Response(200, text=rss_xml)
        if url == "https://good.example/atom":
            return httpx.Response(200, text=atom_xml)
        if url == "https://good.example/empty":
            return httpx.Response(200, text=empty_rss)
        if url == "https://good.example/bozo":
            return httpx.Response(200, text=bozo_rss)
        if url == "https://news.ycombinator.com/":
            return httpx.Response(200, text=html_page)
        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        ok_rss, title_rss, err_rss = await validate_rss_feed_url("https://good.example/rss", client=client)
        assert (ok_rss, title_rss, err_rss) == (True, "AI Weekly", None)

        ok_atom, title_atom, err_atom = await validate_rss_feed_url("https://good.example/atom", client=client)
        assert (ok_atom, title_atom, err_atom) == (True, "Research Atom", None)

        ok_empty, title_empty, err_empty = await validate_rss_feed_url("https://good.example/empty", client=client)
        assert (ok_empty, title_empty, err_empty) == (True, "Empty RSS", None)

        ok_bozo, _, err_bozo = await validate_rss_feed_url("https://good.example/bozo", client=client)
        assert ok_bozo is True and err_bozo is None

        ok_html, _, err_html = await validate_rss_feed_url("https://news.ycombinator.com/", client=client)
        assert ok_html is False and err_html is not None and "valid RSS or Atom" in err_html

        ok_404, _, err_404 = await validate_rss_feed_url("https://good.example/missing", client=client)
        assert ok_404 is False and err_404 is not None

        ok_unsafe, _, err_unsafe = await validate_rss_feed_url("http://127.0.0.1/rss", client=client)
        assert ok_unsafe is False and err_unsafe is not None and "unsafe" in err_unsafe.lower()

        entries, updates = await fetch_new_rss_entries(
            [{"url": "https://news.ycombinator.com/", "etag": None, "last_modified": None}],
            client=client,
        )
        assert entries == []
        assert updates["https://news.ycombinator.com/"]["ok"] is False
        assert "valid RSS or Atom" in updates["https://news.ycombinator.com/"]["error"]
