"""Tests for Settings loading/validation and LLMClient resilience."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import BadRequestError
from pydantic import ValidationError

from config import AGENT_DIR, load_settings, validate_runtime_secrets
from llm import LLMClient, ProfileContext, parse_json_response


def test_load_settings_isolation_clamping_and_validation(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "TELEGRAM_ALLOWED_CHAT_ID=",
                "LLM_API_KEY=sk-test-123",
                "LLM_MAX_CONCURRENCY=6",
                "RSS_FEEDS=https://a.com/rss, https://b.com/rss , https://a.com/rss",
                "MAX_ARTICLES_PER_DIGEST=25",
                "QUIET_HOURS=9-17",
                "DB_PATH=custom/my.db",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)
    assert settings.telegram_bot_token == "12345:TEST_TOKEN"
    assert settings.telegram_allowed_chat_id is None
    assert settings.llm_provider == "openai"
    assert settings.llm_max_concurrency == 6
    assert settings.rss_feeds == ["https://a.com/rss", "https://b.com/rss"]
    assert settings.max_articles_per_digest == 9  # clamped from 25 -> 9
    assert settings.quiet_hours == "09-17"
    assert settings.db_path == AGENT_DIR / "custom/my.db"
    assert settings.feed_max_consecutive_failures == 5
    assert settings.article_retention_days == 30

    # Custom & disabled lifecycle settings
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST",
                "LLM_API_KEY=sk-ok",
                "FEED_MAX_CONSECUTIVE_FAILURES=0",
                "ARTICLE_RETENTION_DAYS=14",
            ]
        ),
        encoding="utf-8",
    )
    lifecycle_settings = load_settings(env_file)
    assert lifecycle_settings.feed_max_consecutive_failures == 0
    assert lifecycle_settings.article_retention_days == 14

    # Negative lifecycle settings rejected
    for bad_line in ("FEED_MAX_CONSECUTIVE_FAILURES=-1", "ARTICLE_RETENTION_DAYS=-5"):
        env_file.write_text(
            f"TELEGRAM_BOT_TOKEN=12345:TEST\nLLM_API_KEY=sk-ok\n{bad_line}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_settings(env_file)

    # Clamping lower bound 0 -> 1
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=12345:TEST\nLLM_API_KEY=sk-ok\nMAX_ARTICLES_PER_DIGEST=0\n",
        encoding="utf-8",
    )
    assert load_settings(env_file).max_articles_per_digest == 1

    # Invalid QUIET_HOURS rejected
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=12345:TEST\nLLM_API_KEY=sk-ok\nQUIET_HOURS=08-08\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_settings(env_file)


def test_codex_provider_settings_and_secret_validation(tmp_path: Path) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(0o755)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "LLM_PROVIDER=codex",
                "LLM_API_KEY=",
                "LLM_MODEL=gpt-6-luna",
                f"CODEX_CLI_PATH={fake_codex}",
                "CODEX_REASONING_EFFORT=low",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    assert settings.llm_provider == "codex"
    assert settings.llm_model == "gpt-6-luna"
    assert settings.codex_reasoning_effort == "low"
    validate_runtime_secrets(settings)

    # Missing executable raises clear ValueError
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST_TOKEN",
                "LLM_PROVIDER=codex",
                f"CODEX_CLI_PATH={tmp_path / 'does_not_exist'}",
            ]
        ),
        encoding="utf-8",
    )
    bad_settings = load_settings(env_file)
    with pytest.raises(ValueError, match="Codex CLI"):
        validate_runtime_secrets(bad_settings)


def test_validate_runtime_secrets_rejects_placeholders(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=12345:TEST\nLLM_API_KEY=REPLACE_WITH_YOUR_LLM_API_KEY\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        validate_runtime_secrets(settings)


def test_parse_json_response_variants() -> None:
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_response('Here is the JSON:\n{"items": []}\nThanks!') == {"items": []}
    assert parse_json_response('<think>consider {broken json} thoughts</think>\n{"ok": true}') == {
        "ok": True
    }
    assert parse_json_response('Note {ignored} prefix {"is_ai": true} suffix') == {"is_ai": True}
    with pytest.raises(ValueError):
        parse_json_response("not json at all")


class _FakeCompletions:
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = self._handler(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeOpenAI:
    def __init__(self, handler: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(handler))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _make_settings(tmp_path: Path) -> Any:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TELEGRAM_BOT_TOKEN=12345:TEST\nLLM_API_KEY=sk-test-real\n",
        encoding="utf-8",
    )
    return load_settings(env_file)


@pytest.mark.asyncio
async def test_llm_bad_request_fallback_and_separate_stage1_calls(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    def handler(kwargs: dict[str, Any]) -> str:
        if "response_format" in kwargs:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(400, request=req)
            raise BadRequestError("response_format is not supported", response=resp, body=None)
        if "temperature" in kwargs:
            req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            resp = httpx.Response(400, request=req)
            raise BadRequestError("temperature not supported", response=resp, body=None)

        user_msg = kwargs["messages"][1]["content"]
        start = user_msg.index("<rss_candidate>") + len("<rss_candidate>")
        end = user_msg.index("</rss_candidate>")
        item = json.loads(user_msg[start:end])
        assert isinstance(item, dict)
        return json.dumps(
            {
                "is_ai": "Title 4" not in item["title"],
                "llm_score": 0.85 if "Title 4" not in item["title"] else 0.1,
                "topics": ["AI Agents & Tool Use"],
                "reason": f"Evaluated {item['title']}.",
            }
        )

    fake = _FakeOpenAI(handler)
    llm = LLMClient(settings, client=fake)  # type: ignore[arg-type]

    candidates = [
        {"id": f"id_{i}", "title": f"Title {i}", "source_feed": "https://hnrss.org/frontpage", "rss_snippet": "AI"}
        for i in range(5)
    ]
    profile = ProfileContext(
        preference_summary="Agents and LLM engineering.",
        top_topics=[("AI Agents & Tool Use", 2.0)],
        known_topics=["AI Agents & Tool Use"],
    )

    results = await llm.score_candidates(candidates, profile)
    assert len(results) == 5
    # Each candidate got its own separate completion flow (5 successful calls + 2 retries per call = 15)
    successful_calls = [c for c in fake.chat.completions.calls if "response_format" not in c and "temperature" not in c]
    assert len(successful_calls) == 5
    assert results["id_0"].id == "id_0"
    assert results["id_0"].is_ai is True
    assert results["id_0"].llm_score == 0.85
    assert results["id_4"].id == "id_4"
    assert results["id_4"].is_ai is False


@pytest.mark.asyncio
async def test_codex_completion_cli_flags_and_output_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(0o755)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=12345:TEST",
                "LLM_PROVIDER=codex",
                "LLM_MODEL=gpt-6-luna",
                f"CODEX_CLI_PATH={fake_codex}",
                "CODEX_REASONING_EFFORT=low",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(env_file)
    recorded_cmds: list[tuple[str, ...]] = []

    class _FakeProc:
        def __init__(self, out_file: Path) -> None:
            self.returncode = 0
            self._out_file = out_file

        async def communicate(self, input_bytes: bytes | None = None) -> tuple[bytes, bytes]:
            assert input_bytes is not None
            assert b"<rss_candidate>" in input_bytes
            self._out_file.write_text(
                json.dumps(
                    {
                        "is_ai": True,
                        "llm_score": 0.92,
                        "topics": ["AI Agents & Tool Use"],
                        "reason": "Strong agent news.",
                    }
                ),
                encoding="utf-8",
            )
            return (b"", b"")

        def kill(self) -> None:
            pass

        async def wait(self) -> int:
            return 0

    async def fake_exec(*cmd: str, **kwargs: Any) -> _FakeProc:
        recorded_cmds.append(cmd)
        out_idx = cmd.index("-o") + 1
        return _FakeProc(Path(cmd[out_idx]))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    llm = LLMClient(settings)
    profile = ProfileContext(preference_summary="Agents")
    verdict = await llm.score_candidate(
        {
            "id": "abc123",
            "title": "Claude Ship Agents",
            "source_feed": "https://hnrss.org/frontpage",
            "rss_snippet": "Code review agents.",
        },
        profile,
    )
    assert verdict.id == "abc123"
    assert verdict.is_ai is True
    assert verdict.llm_score == 0.92

    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert "--skip-git-repo-check" in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-6-luna"
    assert cmd[cmd.index("-c") + 1] == 'model_reasoning_effort="low"'
    assert "--output-schema" in cmd
    assert cmd[-1] == "-"


@pytest.mark.asyncio
async def test_llm_stage1_error_propagates_and_stage2_fallback(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    def failing_handler(_: dict[str, Any]) -> str:
        raise RuntimeError("Upstream timeout")

    llm = LLMClient(settings, client=_FakeOpenAI(failing_handler))  # type: ignore[arg-type]
    profile = ProfileContext(preference_summary="Summary")

    # Stage 1 propagates errors so caller increments stage1_attempts
    with pytest.raises(RuntimeError, match="Upstream timeout"):
        await llm.score_candidates(
            [{"id": "a1", "title": "T", "source_feed": "f", "rss_snippet": "s"}], profile
        )

    # Stage 2 returns deterministic fallback summary instead of crashing digest
    summary = await llm.summarize_article(
        {
            "id": "a1",
            "title": "Cool Paper",
            "url": "https://example.com",
            "source_feed": "f",
            "rss_snippet": "Concrete benchmark speedup of 2x on Llama.",
            "topics_json": '["AI Research & Architectures"]',
        },
        profile,
    )
    assert len(summary.summary_bullets) == 1
    assert "2x on Llama" in summary.summary_bullets[0]
    assert summary.topics == ["AI Research & Architectures"]


@pytest.mark.asyncio
async def test_propose_discovery_topics_dedup(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    def handler(_: dict[str, Any]) -> str:
        return json.dumps(
            {
                "topics": [
                    "mcp",  # already exists case-insensitively
                    "Test-Time Compute",
                    "test-time compute",  # duplicate within batch
                    "Browser-Use Agents",
                ]
            }
        )

    llm = LLMClient(settings, client=_FakeOpenAI(handler))  # type: ignore[arg-type]
    proposed = await llm.propose_discovery_topics(["Title 1"], ["MCP", "AI Coding & DevTools"])
    assert proposed == ["Test-Time Compute", "Browser-Use Agents"]
