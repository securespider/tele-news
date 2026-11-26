"""Tests for academic paper pipeline, BibTeX, FTS5 search, /review, /litreview, /compare & /library."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from telegram.constants import ChatType

import bot as bot_module
import papers as papers_module
from bot import (
    cmd_bibtex,
    cmd_compare,
    cmd_export,
    cmd_library,
    cmd_litreview,
    cmd_note,
    cmd_read,
    cmd_review,
    cmd_save,
    cmd_search,
    cmd_trending,
    run_hf_papers_cycle,
)
from config import load_settings
from db import Database
from learning import CORE_TOPICS
from llm import (
    ComparisonRow,
    LLMClient,
    LiteratureReviewReport,
    PaperComparisonDimension,
    PaperComparisonReport,
    PaperReview,
    Stage1Item,
)
from papers import (
    PaperRecord,
    extract_arxiv_id,
    format_bibtex,
    format_library_markdown,
    format_litreview_markdown,
    paper_record_to_article_dict,
    parse_arxiv_atom,
    parse_hf_daily_papers_payload,
    search_arxiv,
    search_semantic_scholar,
)


def _make_settings(tmp_path: Path) -> Any:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "LLM_API_KEY=sk-test-key",
                "TELEGRAM_ADMIN_IDS=1001",
                "ENABLE_HF_DAILY_PAPERS=true",
                "ENABLE_ARXIV_API=true",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env_file)


def test_arxiv_id_extraction_and_bibtex_latex_escaping() -> None:
    assert extract_arxiv_id("2501.12345") == "2501.12345"
    assert extract_arxiv_id("arXiv:2405.09876v2") == "2405.09876v2"
    assert extract_arxiv_id("https://arxiv.org/pdf/2312.00001.pdf") == "2312.00001"
    assert extract_arxiv_id("https://huggingface.co/papers/2502.54321") == "2502.54321"
    assert extract_arxiv_id("cs/0102003") == "cs/0102003"
    assert extract_arxiv_id("not an arxiv id") is None

    rec = PaperRecord(
        id="p1",
        arxiv_id="2501.12345",
        title="Scaling 100% Sparse_Attention & MoE #1",
        authors=["Alice Smith", "Bob Jones"],
        published_at="2025-01-15T00:00:00+00:00",
        venue="NeurIPS 2025",
        url="https://arxiv.org/abs/2501.12345",
    )
    bib = format_bibtex(rec)
    assert "@inproceedings{smith2025scaling," in bib
    assert r"100\% Sparse\_Attention \& MoE \#1" in bib
    assert "Alice Smith and Bob Jones" in bib
    assert "eprint = {2501.12345}" in bib
    assert "archivePrefix = {arXiv}" in bib


def test_parse_hf_daily_papers_and_arxiv_atom() -> None:
    hf_payload = [
        {
            "title": "Test-Time Compute Agents",
            "publishedAt": "2026-11-25T10:00:00Z",
            "paper": {
                "id": "2611.00123",
                "title": "Test-Time Compute Agents",
                "summary": "We show 3x gains. Code at https://github.com/org/ttc-agents.",
                "authors": [{"name": "Jane Doe"}, {"name": "John Roe"}],
                "upvotes": 42,
            },
        }
    ]
    parsed_hf = parse_hf_daily_papers_payload(hf_payload)
    assert len(parsed_hf) == 1
    assert parsed_hf[0]["arxiv_id"] == "2611.00123"
    assert parsed_hf[0]["hf_upvotes"] == 42
    assert parsed_hf[0]["code_url"] == "https://github.com/org/ttc-agents"
    assert parsed_hf[0]["content_type"] == "paper"

    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2611.00999v1</id>
        <published>2026-11-20T00:00:00Z</published>
        <title>Fast Speculative Decoding in C++</title>
        <summary>Delivers 2.4x speedup. Repo: https://github.com/ai/spec-dec</summary>
        <author><name>Ada Lovelace</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/2611.00999v1"/>
        <link title="pdf" href="https://arxiv.org/pdf/2611.00999v1"/>
      </entry>
    </feed>"""
    atom_recs = parse_arxiv_atom(atom_xml)
    assert len(atom_recs) == 1
    assert atom_recs[0].arxiv_id == "2611.00999v1"
    assert atom_recs[0].authors == ["Ada Lovelace"]
    assert atom_recs[0].code_url == "https://github.com/ai/spec-dec"


@pytest.mark.asyncio
async def test_arxiv_and_semantic_scholar_graceful_429_fallback() -> None:
    def s2_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Rate limited")

    async with httpx.AsyncClient(transport=httpx.MockTransport(s2_handler)) as client:
        res = await search_semantic_scholar("speculative decoding", client=client)
        assert res == []

    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2611.00555v1</id>
        <published>2026-11-21T00:00:00Z</published>
        <title>Grounded RAG Verification</title>
        <summary>Reduces hallucinations by 40%.</summary>
        <author><name>Grace Hopper</name></author>
      </entry>
    </feed>"""

    def arxiv_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=atom_xml)

    async with httpx.AsyncClient(transport=httpx.MockTransport(arxiv_handler)) as client:
        recs = await search_arxiv("RAG", client=client, throttle_seconds=0.0)
        assert len(recs) == 1
        assert recs[0].title == "Grounded RAG Verification"


@pytest.mark.asyncio
async def test_llm_review_litreview_compare_and_fallbacks(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    class _FakeCompletions:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.calls = 0

        async def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            if self.fail:
                raise RuntimeError("Upstream 503")
            sys_msg = kwargs["messages"][0]["content"]
            if "peer reviewer" in sys_msg:
                content = json.dumps(
                    {
                        "tldr": "Introduces sparse KV eviction with 2x throughput.",
                        "problem": "KV cache memory bottleneck in long-context LLMs.",
                        "methodology": "Attention-sink + heavy-hitter dynamic pruning.",
                        "key_results": ["2.1x decode throughput at 128k context"],
                        "limitations": ["Evaluated only on Llama-3 8B"],
                        "practical_takeaways": "Useful for memory-bound batch serving.",
                        "maturity_level": "engineering_ready",
                        "reproducibility_notes": "MIT-licensed code released.",
                        "topics": ["KV Cache", "AI Hardware, Chips & Inference"],
                    }
                )
            elif "systematic literature review" in sys_msg:
                content = json.dumps(
                    {
                        "executive_summary": "Both [P1] and [P2] improve long-context serving.",
                        "consensus_and_debates": "[P1] prunes KV while [P2] quantizes weights.",
                        "comparison_rows": [
                            {
                                "paper_ref": "[P1]",
                                "method": "KV Pruning",
                                "benchmark_or_data": "LongBench",
                                "key_finding": "2.1x speedup",
                                "limitation": "8B model only",
                            }
                        ],
                        "research_gaps": ["Combining KV pruning with FP4 weights"],
                        "recommended_reading_order": ["[P1] Start here for KV memory wins"],
                    }
                )
            else:
                content = json.dumps(
                    {
                        "summary": "P1 targets KV cache; P2 targets decoding speed.",
                        "dimensions": [
                            {"dimension": "Mechanism", "assessments": ["KV eviction", "Draft model"]},
                        ],
                        "verdict_for_practitioners": "Use P1 for memory, P2 for latency.",
                    }
                )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(fail=False)),
        close=lambda: asyncio.sleep(0),
    )
    llm = LLMClient(settings, client=fake_client)  # type: ignore[arg-type]
    p1 = {"id": "p1", "title": "Paper 1", "arxiv_id": "2501.00001", "rss_snippet": "KV cache pruning."}
    p2 = {"id": "p2", "title": "Paper 2", "arxiv_id": "2501.00002", "rss_snippet": "Speculative decoding."}

    review = await llm.review_paper(p1)
    assert review.maturity_level == "engineering_ready"
    assert "2.1x" in review.key_results[0]

    lit = await llm.synthesize_literature_review("long context inference", [p1, p2])
    assert "[P1]" in lit.executive_summary
    md = format_litreview_markdown("long context inference", lit, [p1, p2])
    assert "## Paper Comparison Matrix" in md
    assert "https://arxiv.org/abs/2501.00001" in format_library_markdown([
        {**p1, "url": "https://arxiv.org/abs/2501.00001", "status": "to_read", "tags_json": '["kv"]', "notes": "Great"}
    ])

    cmp_rep = await llm.compare_papers([p1, p2])
    assert len(cmp_rep.dimensions) == 1

    # Fallback mode on LLM outage never crashes
    fail_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(fail=True)),
        close=lambda: asyncio.sleep(0),
    )
    llm_fail = LLMClient(settings, client=fail_client)  # type: ignore[arg-type]
    assert "fallback" in (await llm_fail.review_paper(p1)).limitations[0].lower()
    assert "[P1]" in (await llm_fail.synthesize_literature_review("kv", [p1])).recommended_reading_order[0]
    assert len((await llm_fail.compare_papers([p1, p2])).dimensions) == 2


class _FakeReplyMsg:
    def __init__(self, text: str, replies: list[str], docs: list[str]) -> None:
        self.text = text
        self._replies = replies
        self._docs = docs

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self._replies.append(text)

    async def reply_document(self, document: Any, **kwargs: Any) -> None:
        self._docs.append(document.filename)


@pytest.mark.asyncio
async def test_fts5_search_review_caching_litreview_compare_and_library_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(tmp_path)
    db = Database(tmp_path / "lit.db")
    db.init_schema()
    db.seed_core_topics(CORE_TOPICS)
    db.upsert_user(1001, username="lead", role="admin", status="active", core_topics=CORE_TOPICS)
    db.upsert_user(1002, username="researcher", role="member", status="active", core_topics=CORE_TOPICS)

    p1 = paper_record_to_article_dict(
        PaperRecord(
            id="paper_1",
            arxiv_id="2611.00001",
            title="C++ Speculative Decoding Kernels",
            authors=["Alice Researcher"],
            abstract="Fast C++ kernels for LLM speculative decoding.",
            url="https://arxiv.org/abs/2611.00001",
            hf_upvotes=55,
            citation_count=12,
        )
    )
    p2 = paper_record_to_article_dict(
        PaperRecord(
            id="paper_2",
            arxiv_id="2611.00002",
            title="Ring Attention for 1M Context Windows",
            authors=["Bob Engineer"],
            abstract="Distributed attention across GPUs.",
            url="https://arxiv.org/abs/2611.00002",
            hf_upvotes=30,
        )
    )
    db.upsert_paper_article(p1)
    db.upsert_paper_article(p2)

    # FTS5 safely handles special characters like C++
    hits = db.search_articles_fts("C++ decoding!")
    assert len(hits) == 1
    assert hits[0]["id"] == "paper_1"

    class _MockResearchLLM:
        def __init__(self) -> None:
            self.review_calls = 0

        async def score_candidate(self, candidate: dict[str, Any], _profile: Any) -> Stage1Item:
            return Stage1Item(id=candidate["id"], is_ai=True, llm_score=0.9, topics=["AI Research & Architectures"], reason="High signal.")

        async def review_paper(self, paper: dict[str, Any], full_text: str | None = None) -> PaperReview:
            self.review_calls += 1
            return PaperReview(
                tldr="Fast C++ kernels.",
                problem="Latency.",
                methodology="Draft + verify.",
                key_results=["2.5x speedup"],
                limitations=["GPU specific"],
                practical_takeaways="Deploy for batch=1.",
                maturity_level="engineering_ready",
                reproducibility_notes="Open source.",
                topics=["AI Hardware, Chips & Inference"],
            )

        async def synthesize_literature_review(self, query: str, papers: list[dict[str, Any]]) -> LiteratureReviewReport:
            return LiteratureReviewReport(
                executive_summary="Synthesis across [P1] and [P2].",
                consensus_and_debates="Both reduce serving cost.",
                comparison_rows=[
                    ComparisonRow(paper_ref="[P1]", method="Drafting", benchmark_or_data="MT-Bench", key_finding="2.5x", limitation="Batch 1")
                ],
                research_gaps=["Combined kernel"],
                recommended_reading_order=["[P1] first"],
            )

        async def compare_papers(self, papers: list[dict[str, Any]]) -> PaperComparisonReport:
            return PaperComparisonReport(
                summary="P1 optimizes latency; P2 optimizes context.",
                dimensions=[PaperComparisonDimension(dimension="Goal", assessments=["Latency", "Context"])],
                verdict_for_practitioners="Combine both.",
            )

    llm = _MockResearchLLM()
    replies: list[str] = []
    docs: list[str] = []
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
        upd = SimpleNamespace(
            poll_answer=None,
            effective_user=SimpleNamespace(id=uid, username=f"u{uid}", first_name="User", last_name=""),
            effective_chat=SimpleNamespace(id=uid, type=ChatType.PRIVATE),
            effective_message=_FakeReplyMsg(text, replies, docs),
        )
        ctx = SimpleNamespace(application=app, bot=SimpleNamespace(), args=args or [])
        return upd, ctx

    # User 1001 runs /review -> LLM called once & cached
    await cmd_review(*mk(1001, "/review", ["2611.00001"]))
    assert llm.review_calls == 1
    assert any("Paper Review" in r for r in replies)

    # User 1002 runs /review on same paper -> served from shared cache (0 extra LLM calls!)
    await cmd_review(*mk(1002, "/review", ["2611.00001"]))
    assert llm.review_calls == 1

    # /litreview generates HTML + litreview.md + references.bib
    monkeypatch.setattr(bot_module, "search_arxiv", lambda *a, **kw: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(bot_module, "search_semantic_scholar", lambda *a, **kw: asyncio.sleep(0, result=[]))
    await cmd_litreview(*mk(1001, "/litreview", ["decoding", "attention"]))
    assert "litreview.md" in docs
    assert "references.bib" in docs

    # /compare works across 2 papers
    await cmd_compare(*mk(1001, "/compare", ["2611.00001", "2611.00002"]))
    assert any("Side-by-Side Paper Comparison" in r for r in replies)

    # /search finds local FTS hit
    await cmd_search(*mk(1001, "/search", ["Speculative"]))
    assert any("Speculative Decoding" in r for r in replies)

    # /save, /note, /read, /library, /bibtex, /export
    await cmd_save(*mk(1001, "/save", ["2611.00001", "inference,kernels"]))
    await cmd_note(*mk(1001, "/note", ["1", "Benchmark on H100"]))
    await cmd_read(*mk(1001, "/read", ["1"]))
    await cmd_library(*mk(1001, "/library"))
    assert any("Benchmark on H100" in r for r in replies)

    await cmd_bibtex(*mk(1001, "/bibtex", ["2611.00001"]))
    assert any("eprint = {2611.00001}" in r for r in replies)

    await cmd_export(*mk(1001, "/export", ["bibtex"]))
    await cmd_export(*mk(1001, "/export", ["markdown"]))
    assert "library.bib" in docs
    assert "library.md" in docs

    # /trending & HF papers cycle
    monkeypatch.setattr(
        bot_module,
        "fetch_hf_daily_papers",
        lambda limit=25: asyncio.sleep(0, result=[p1]),
    )
    hf_summary = await run_hf_papers_cycle(db, llm, settings)  # type: ignore[arg-type]
    assert hf_summary["fetched"] == 1
    await cmd_trending(*mk(1001, "/trending"))
    assert any("Hugging Face Daily Papers" in r for r in replies)
