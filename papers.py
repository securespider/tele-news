"""Academic paper ingestion, search, enrichment, BibTeX & Markdown export."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import httpx

from scraper import RSS_USER_AGENT, compute_article_id, extract_github_urls
from security import create_safe_async_client, safe_fetch_bytes, safe_fetch_text

logger = logging.getLogger(__name__)

_ARXIV_NEW_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_ARXIV_OLD_ID_RE = re.compile(r"\b([a-z-]+/\d{7}(?:v\d+)?)\b", re.IGNORECASE)
_ARXIV_PREFIX_RE = re.compile(
    r"(?:arxiv:\s*|arxiv\.org/(?:abs|pdf)/|huggingface\.co/papers/)"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_LATEX_SPECIAL_CHARS = str.maketrans(
    {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
)
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_ARXIV_THROTTLE_LOCK: asyncio.Lock | None = None
_ARXIV_LAST_REQ_TS: float = 0.0


@dataclass
class PaperRecord:
    """Normalized academic paper metadata across arXiv, Hugging Face, and Semantic Scholar."""

    id: str
    arxiv_id: str | None = None
    doi: str | None = None
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    url: str = ""
    pdf_url: str | None = None
    published_at: str | None = None
    venue: str | None = None
    citation_count: int | None = None
    influential_citations: int | None = None
    hf_upvotes: int | None = None
    code_url: str | None = None
    source: str = "arxiv"


def extract_arxiv_id(text: str | None) -> str | None:
    """Extract canonical arXiv identifier from IDs, arXiv URLs, or HF paper URLs."""
    raw = (text or "").strip()
    if not raw:
        return None
    prefixed = _ARXIV_PREFIX_RE.search(raw)
    if prefixed:
        return prefixed.group(1).removesuffix(".pdf")
    new_match = _ARXIV_NEW_ID_RE.search(raw)
    if new_match:
        return new_match.group(1)
    old_match = _ARXIV_OLD_ID_RE.search(raw)
    if old_match:
        return old_match.group(1)
    return None


def _escape_bibtex_text(value: str) -> str:
    cleaned = " ".join((value or "").split())
    return cleaned.translate(_LATEX_SPECIAL_CHARS)


def _extract_year(published_at: str | None) -> str:
    if published_at:
        match = re.match(r"^(20\d{2}|19\d{2})", str(published_at).strip())
        if match:
            return match.group(1)
    return str(datetime.now(timezone.utc).year)


def _build_bibtex_key(authors: list[str], year: str, title: str) -> str:
    first_surname = "paper"
    if authors:
        parts = re.findall(r"[A-Za-z]+", authors[0])
        if parts:
            first_surname = parts[-1].lower()
    stopwords = {"a", "an", "the", "of", "in", "on", "for", "to", "with", "via", "using"}
    title_words = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]+", title)
        if w.lower() not in stopwords
    ]
    keyword = title_words[0] if title_words else "work"
    return re.sub(r"[^a-z0-9]", "", f"{first_surname}{year}{keyword}") or f"ref{year}"


def format_bibtex(record: dict[str, Any] | PaperRecord) -> str:
    """Generate a clean, LaTeX-escaped BibTeX citation entry from a PaperRecord or article dict."""
    if isinstance(record, PaperRecord):
        title = record.title
        authors = record.authors
        published_at = record.published_at
        arxiv_id = record.arxiv_id
        doi = record.doi
        venue = record.venue
        url = record.url
    else:
        title = str(record.get("title") or "Untitled")
        raw_authors = record.get("authors")
        if isinstance(raw_authors, list):
            authors = [str(a) for a in raw_authors if str(a).strip()]
        else:
            authors_json = record.get("authors_json")
            authors = []
            if authors_json:
                try:
                    loaded = json.loads(authors_json)
                    if isinstance(loaded, list):
                        authors = [str(a) for a in loaded if str(a).strip()]
                except (TypeError, ValueError):
                    pass
        published_at = record.get("published_at")
        arxiv_id = record.get("arxiv_id") or extract_arxiv_id(record.get("url"))
        doi = record.get("doi")
        venue = record.get("venue")
        url = str(record.get("url") or "")

    year = _extract_year(published_at)
    cite_key = _build_bibtex_key(authors, year, title)
    authors_str = " and ".join(_escape_bibtex_text(a) for a in authors) if authors else "Unknown Author"
    escaped_title = _escape_bibtex_text(title)

    entry_type = "inproceedings" if venue and "arxiv" not in venue.lower() else ("article" if arxiv_id else "misc")
    fields: list[tuple[str, str]] = [
        ("title", f"{{{escaped_title}}}"),
        ("author", f"{{{authors_str}}}"),
        ("year", f"{{{year}}}"),
    ]
    if venue:
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append((venue_field, f"{{{_escape_bibtex_text(venue)}}}"))
    elif arxiv_id:
        fields.append(("journal", f"{{arXiv preprint arXiv:{arxiv_id}}}"))

    if arxiv_id:
        fields.append(("eprint", f"{{{arxiv_id}}}"))
        fields.append(("archivePrefix", "{arXiv}"))
    if doi:
        fields.append(("doi", f"{{{doi}}}"))
    if url:
        fields.append(("url", f"{{{url}}}"))

    lines = [f"@{entry_type}{{{cite_key},"]
    for idx, (k, v) in enumerate(fields):
        comma = "," if idx < len(fields) - 1 else ""
        lines.append(f"  {k} = {v}{comma}")
    lines.append("}")
    return "\n".join(lines)


def paper_record_to_article_dict(paper: PaperRecord) -> dict[str, Any]:
    """Convert a PaperRecord into an articles-table compatible dict."""
    url = paper.url or (f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else "")
    return {
        "id": paper.id or compute_article_id(url),
        "url": url,
        "comments_url": None,
        "title": paper.title,
        "source_feed": f"https://arxiv.org" if paper.source == "arxiv" else paper.source,
        "published_at": paper.published_at or datetime.now(timezone.utc).isoformat(),
        "rss_snippet": paper.abstract[:800],
        "full_text": paper.abstract,
        "content_type": "paper",
        "arxiv_id": paper.arxiv_id,
        "authors_json": json.dumps(paper.authors),
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "hf_upvotes": paper.hf_upvotes,
        "code_url": paper.code_url,
        "bibtex": format_bibtex(paper),
    }


def parse_hf_daily_papers_payload(raw_json: Any, *, limit: int = 25) -> list[dict[str, Any]]:
    """Parse JSON list from Hugging Face Daily Papers API into normalized paper dicts."""
    if not isinstance(raw_json, list):
        return []

    results: list[dict[str, Any]] = []
    for item in raw_json[:limit]:
        if not isinstance(item, dict):
            continue
        paper_obj = item.get("paper") if isinstance(item.get("paper"), dict) else item
        arxiv_id = str(paper_obj.get("id") or item.get("id") or "").strip()
        title = " ".join(str(item.get("title") or paper_obj.get("title") or "").split())
        if not arxiv_id or not title:
            continue

        abstract = " ".join(
            str(paper_obj.get("summary") or item.get("summary") or "").split()
        )
        authors: list[str] = []
        for a in paper_obj.get("authors") or []:
            if isinstance(a, dict):
                name = str(a.get("name") or "").strip()
            else:
                name = str(a or "").strip()
            if name and name not in authors:
                authors.append(name)

        upvotes_raw = paper_obj.get("upvotes") if "upvotes" in paper_obj else item.get("upvotes")
        try:
            upvotes = int(upvotes_raw) if upvotes_raw is not None else None
        except (TypeError, ValueError):
            upvotes = None

        code_url = str(paper_obj.get("githubRepo") or "").strip() or None
        if not code_url:
            gh = extract_github_urls(abstract)
            code_url = gh[0] if gh else None

        published_at = str(
            paper_obj.get("publishedAt") or item.get("publishedAt") or datetime.now(timezone.utc).isoformat()
        )
        url = f"https://arxiv.org/abs/{arxiv_id}"
        rec = PaperRecord(
            id=compute_article_id(url),
            arxiv_id=arxiv_id,
            title=title,
            authors=authors[:20],
            abstract=abstract,
            url=url,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            published_at=published_at,
            hf_upvotes=upvotes,
            code_url=code_url,
            source="https://huggingface.co/papers",
        )
        art = paper_record_to_article_dict(rec)
        art["source_feed"] = "https://huggingface.co/papers"
        results.append(art)
    return results


async def fetch_hf_daily_papers(
    *,
    client: httpx.AsyncClient | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Fetch trending daily AI papers from Hugging Face Daily Papers public API."""
    api_url = "https://huggingface.co/api/daily_papers"
    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )
    try:
        _, _, _, raw_bytes = await safe_fetch_bytes(
            active_client, api_url, max_bytes=5 * 1024 * 1024
        )
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        return parse_hf_daily_papers_payload(data, limit=limit)
    except Exception as exc:
        logger.warning("Failed fetching Hugging Face Daily Papers: %s", exc)
        return []
    finally:
        if owns_client:
            await active_client.aclose()


def parse_arxiv_atom(xml_text: str) -> list[PaperRecord]:
    """Parse arXiv Atom XML response into PaperRecord objects."""
    if not (xml_text or "").strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("Invalid arXiv Atom XML: %s", exc)
        return []

    records: list[PaperRecord] = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        arxiv_id = extract_arxiv_id(raw_id)
        title = " ".join(
            (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").split()
        )
        if not title or title.lower() == "error":
            continue

        abstract = " ".join(
            (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").split()
        )
        published_at = (
            entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
        ).strip() or datetime.now(timezone.utc).isoformat()

        authors: list[str] = []
        for author_el in entry.findall("atom:author", _ATOM_NS):
            name = (author_el.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
        for link_el in entry.findall("atom:link", _ATOM_NS):
            href = (link_el.attrib.get("href") or "").strip()
            if link_el.attrib.get("title") == "pdf" and href:
                pdf_url = href
            elif link_el.attrib.get("rel") == "alternate" and href:
                abs_url = href

        if not abs_url:
            continue
        code_urls = extract_github_urls(abstract)

        records.append(
            PaperRecord(
                id=compute_article_id(abs_url),
                arxiv_id=arxiv_id,
                title=title,
                authors=authors[:20],
                abstract=abstract,
                url=abs_url,
                pdf_url=pdf_url,
                published_at=published_at,
                code_url=code_urls[0] if code_urls else None,
                source="arxiv",
            )
        )
    return records


async def _respect_arxiv_throttle(throttle_seconds: float) -> None:
    global _ARXIV_THROTTLE_LOCK, _ARXIV_LAST_REQ_TS
    if throttle_seconds <= 0:
        return
    if _ARXIV_THROTTLE_LOCK is None:
        _ARXIV_THROTTLE_LOCK = asyncio.Lock()
    async with _ARXIV_THROTTLE_LOCK:
        elapsed = time.monotonic() - _ARXIV_LAST_REQ_TS
        if elapsed < throttle_seconds:
            await asyncio.sleep(throttle_seconds - elapsed)
        _ARXIV_LAST_REQ_TS = time.monotonic()


async def search_arxiv(
    query: str,
    *,
    max_results: int = 8,
    categories: list[str] | None = None,
    client: httpx.AsyncClient | None = None,
    throttle_seconds: float = 3.0,
) -> list[PaperRecord]:
    """Query the arXiv Atom API safely with rate throttling."""
    cleaned_q = " ".join((query or "").split())
    if not cleaned_q:
        return []

    terms = [f'all:"{cleaned_q}"'] if " " in cleaned_q else [f"all:{cleaned_q}"]
    if categories:
        cat_expr = " OR ".join(f"cat:{c.strip()}" for c in categories if c.strip())
        if cat_expr:
            search_expr = f"({terms[0]}) AND ({cat_expr})"
        else:
            search_expr = terms[0]
    else:
        search_expr = terms[0]

    api_url = (
        "https://export.arxiv.org/api/query"
        f"?search_query={quote_plus(search_expr)}"
        f"&start=0&max_results={max(1, min(20, max_results))}"
        "&sortBy=relevance&sortOrder=descending"
    )

    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )
    try:
        await _respect_arxiv_throttle(throttle_seconds)
        _, _, _, xml_text = await safe_fetch_text(
            active_client, api_url, max_bytes=3 * 1024 * 1024
        )
        return parse_arxiv_atom(xml_text)
    except Exception as exc:
        logger.warning("arXiv API query failed (%s): %s", cleaned_q[:60], exc)
        return []
    finally:
        if owns_client:
            await active_client.aclose()


async def fetch_arxiv_paper(
    arxiv_id: str,
    *,
    client: httpx.AsyncClient | None = None,
    throttle_seconds: float = 3.0,
) -> PaperRecord | None:
    """Fetch a single paper by arXiv ID via the arXiv Atom API."""
    cleaned = extract_arxiv_id(arxiv_id) or arxiv_id.strip()
    if not cleaned:
        return None
    api_url = f"https://export.arxiv.org/api/query?id_list={quote_plus(cleaned)}"

    owns_client = client is None
    active_client = client or create_safe_async_client(
        timeout=15.0,
        headers={"User-Agent": RSS_USER_AGENT},
    )
    try:
        await _respect_arxiv_throttle(throttle_seconds)
        _, _, _, xml_text = await safe_fetch_text(
            active_client, api_url, max_bytes=2 * 1024 * 1024
        )
        records = parse_arxiv_atom(xml_text)
        return records[0] if records else None
    except Exception as exc:
        logger.warning("arXiv paper fetch failed for %s: %s", cleaned, exc)
        return None
    finally:
        if owns_client:
            await active_client.aclose()


def _s2_dict_to_paper_record(item: dict[str, Any]) -> PaperRecord | None:
    title = " ".join(str(item.get("title") or "").split())
    if not title:
        return None
    ext_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arXiv")
    doi = ext_ids.get("DOI")
    abs_url = (
        f"https://arxiv.org/abs/{arxiv_id}"
        if arxiv_id
        else str(item.get("url") or "").strip()
    )
    if not abs_url:
        return None

    authors = [
        str(a.get("name") or "").strip()
        for a in (item.get("authors") or [])
        if isinstance(a, dict) and str(a.get("name") or "").strip()
    ]
    abstract = " ".join(str(item.get("abstract") or "").split())
    year = item.get("year")
    published_at = f"{year}-01-01T00:00:00+00:00" if year else None
    open_pdf = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
    pdf_url = open_pdf.get("url") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None)
    code_urls = extract_github_urls(abstract)

    return PaperRecord(
        id=compute_article_id(abs_url),
        arxiv_id=str(arxiv_id) if arxiv_id else None,
        doi=str(doi) if doi else None,
        title=title,
        authors=authors[:20],
        abstract=abstract,
        url=abs_url,
        pdf_url=pdf_url,
        published_at=published_at,
        venue=str(item.get("venue") or "").strip() or None,
        citation_count=item.get("citationCount"),
        influential_citations=item.get("influentialCitationCount"),
        code_url=code_urls[0] if code_urls else None,
        source="semantic_scholar",
    )


async def fetch_semantic_scholar_paper(
    identifier: str,
    *,
    api_key: str = "",
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Enrich a paper with citation counts & venue from Semantic Scholar (best-effort)."""
    cleaned = (identifier or "").strip()
    if not cleaned:
        return None
    arxiv_id = extract_arxiv_id(cleaned)
    lookup = f"ARXIV:{re.sub(r'v\\d+$', '', arxiv_id)}" if arxiv_id else cleaned
    fields = "paperId,title,abstract,year,venue,citationCount,influentialCitationCount,authors,externalIds,url,openAccessPdf"
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{quote_plus(lookup)}?fields={fields}"
    headers = {"User-Agent": RSS_USER_AGENT}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()

    owns_client = client is None
    active_client = client or create_safe_async_client(timeout=10.0, headers=headers)
    try:
        _, _, _, raw_bytes = await safe_fetch_bytes(
            active_client, api_url, headers=headers, max_bytes=2 * 1024 * 1024
        )
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("Semantic Scholar enrichment unavailable for %s: %s", lookup, exc)
        return None
    finally:
        if owns_client:
            await active_client.aclose()


async def search_semantic_scholar(
    query: str,
    *,
    limit: int = 6,
    api_key: str = "",
    client: httpx.AsyncClient | None = None,
) -> list[PaperRecord]:
    """Search Semantic Scholar paper graph API (best-effort)."""
    cleaned = " ".join((query or "").split())
    if not cleaned:
        return []
    fields = "paperId,title,abstract,year,venue,citationCount,influentialCitationCount,authors,externalIds,url,openAccessPdf"
    api_url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote_plus(cleaned)}&limit={max(1, min(15, limit))}&fields={fields}"
    )
    headers = {"User-Agent": RSS_USER_AGENT}
    if api_key.strip():
        headers["x-api-key"] = api_key.strip()

    owns_client = client is None
    active_client = client or create_safe_async_client(timeout=10.0, headers=headers)
    try:
        _, _, _, raw_bytes = await safe_fetch_bytes(
            active_client, api_url, headers=headers, max_bytes=2 * 1024 * 1024
        )
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        records: list[PaperRecord] = []
        for item in data.get("data") or []:
            if isinstance(item, dict):
                rec = _s2_dict_to_paper_record(item)
                if rec is not None:
                    records.append(rec)
        return records
    except Exception as exc:
        logger.debug("Semantic Scholar search failed for %s: %s", cleaned[:60], exc)
        return []
    finally:
        if owns_client:
            await active_client.aclose()


async def resolve_paper(
    query_or_id: str,
    *,
    api_key: str = "",
    client: httpx.AsyncClient | None = None,
    throttle_seconds: float = 3.0,
) -> PaperRecord | None:
    """Resolve an arXiv ID/URL or title query via arXiv + Semantic Scholar enrichment."""
    arxiv_id = extract_arxiv_id(query_or_id)
    paper: PaperRecord | None = None
    if arxiv_id:
        paper = await fetch_arxiv_paper(
            arxiv_id, client=client, throttle_seconds=throttle_seconds
        )
    else:
        matches = await search_arxiv(
            query_or_id, max_results=1, client=client, throttle_seconds=throttle_seconds
        )
        paper = matches[0] if matches else None

    if paper is not None:
        s2_lookup = paper.arxiv_id or paper.title
        s2_data = await fetch_semantic_scholar_paper(
            s2_lookup, api_key=api_key, client=client
        )
        if s2_data:
            if s2_data.get("citationCount") is not None:
                paper.citation_count = int(s2_data["citationCount"])
            if s2_data.get("influentialCitationCount") is not None:
                paper.influential_citations = int(s2_data["influentialCitationCount"])
            if s2_data.get("venue"):
                paper.venue = str(s2_data["venue"]).strip() or paper.venue
            ext_ids = s2_data.get("externalIds") or {}
            if isinstance(ext_ids, dict) and ext_ids.get("DOI"):
                paper.doi = str(ext_ids["DOI"])
        return paper

    s2_matches = await search_semantic_scholar(
        query_or_id, limit=1, api_key=api_key, client=client
    )
    return s2_matches[0] if s2_matches else None


def format_library_markdown(items: list[dict[str, Any]], org_name: str = "DailyAI") -> str:
    """Export a user's reading library as a clean Markdown document."""
    lines = [
        f"# {org_name} — Personal Research Library",
        "",
        f"Exported: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total saved items: {len(items)}",
        "",
        "| # | Status | Type | Title | Meta | Tags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "Untitled").replace("|", "\\|")
        url = str(item.get("url") or "")
        status = str(item.get("status") or "to_read")
        ctype = str(item.get("content_type") or "news")
        meta_parts: list[str] = []
        if item.get("arxiv_id"):
            meta_parts.append(f"arXiv:{item['arxiv_id']}")
        if item.get("citation_count") is not None:
            meta_parts.append(f"{item['citation_count']} cites")
        if item.get("hf_upvotes") is not None:
            meta_parts.append(f"{item['hf_upvotes']} HF▲")
        tags = []
        try:
            tags = json.loads(item.get("tags_json") or "[]")
        except (TypeError, ValueError):
            pass
        tags_str = ", ".join(str(t) for t in tags).replace("|", "\\|") or "-"
        meta_str = ", ".join(meta_parts) or "-"
        lines.append(
            f"| {idx} | {status} | {ctype} | [{title}]({url}) | {meta_str} | {tags_str} |"
        )

    notes_items = [it for it in items if (it.get("notes") or "").strip()]
    if notes_items:
        lines.extend(["", "## Reading Notes", ""])
        for it in notes_items:
            lines.append(f"### {it.get('title', 'Untitled')}")
            lines.append(f"- URL: {it.get('url', '')}")
            lines.append(f"- Notes: {it['notes']}")
            lines.append("")

    return "\n".join(lines)


def format_litreview_markdown(
    query: str,
    report: Any,
    papers: list[dict[str, Any]],
) -> str:
    """Export a LiteratureReviewReport and grounded paper references as Markdown."""
    lines = [
        f"# Literature Review: {query}",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Grounded Corpus: {len(papers)} papers",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Consensus & Debates",
        report.consensus_and_debates,
        "",
        "## Paper Comparison Matrix",
        "",
        "| Ref | Method | Benchmark / Data | Key Finding | Limitation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.comparison_rows:
        ref = str(row.paper_ref).replace("|", "\\|")
        method = str(row.method).replace("|", "\\|")
        data = str(row.benchmark_or_data).replace("|", "\\|")
        finding = str(row.key_finding).replace("|", "\\|")
        limitation = str(row.limitation).replace("|", "\\|")
        lines.append(f"| {ref} | {method} | {data} | {finding} | {limitation} |")

    if report.research_gaps:
        lines.extend(["", "## Open Research Gaps"])
        for gap in report.research_gaps:
            lines.append(f"- {gap}")

    if report.recommended_reading_order:
        lines.extend(["", "## Recommended Reading Order"])
        for idx, step in enumerate(report.recommended_reading_order, start=1):
            lines.append(f"{idx}. {step}")

    lines.extend(["", "## Grounded References", ""])
    for idx, paper in enumerate(papers, start=1):
        title = paper.get("title") or "Untitled"
        url = paper.get("url") or ""
        arxiv_tag = f" (arXiv:{paper['arxiv_id']})" if paper.get("arxiv_id") else ""
        lines.append(f"- **[P{idx}]** [{title}]({url}){arxiv_tag}")

    return "\n".join(lines)
