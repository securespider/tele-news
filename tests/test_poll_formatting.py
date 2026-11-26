"""Tests for Telegram poll option guardrails and question truncation."""

from __future__ import annotations

from learning import build_safe_poll_options, truncate_question


def test_build_safe_poll_options_zero_items_returns_empty() -> None:
    options, metadata = build_safe_poll_options([])
    assert options == []
    assert metadata == []


def test_build_safe_poll_options_counts_and_bounds() -> None:
    for count in (1, 5, 15):
        items = [(f"AI Story Title Number {i}", {"article_id": f"id_{i}"}) for i in range(count)]
        options, metadata = build_safe_poll_options(items)

        assert 2 <= len(options) <= 10
        assert len(options) == len(metadata)
        assert options[-1] == "None of these / More variety"
        assert metadata[-1] == {"none_option": True}

        expected_primary = min(count, 9)
        assert len(options) == expected_primary + 1
        for opt in options:
            assert 1 <= len(opt) <= 100


def test_build_safe_poll_options_dedup_long_colliding_labels_stays_under_100() -> None:
    long_prefix = "A" * 110
    items = [
        (long_prefix + " first", {"article_id": "id_1"}),
        (long_prefix + " second", {"article_id": "id_2"}),
        (long_prefix + " third", {"article_id": "id_3"}),
        ("", {"article_id": "id_4"}),
    ]
    options, metadata = build_safe_poll_options(items)

    assert len(options) == 5
    assert len(set(options)) == 5
    for opt in options:
        assert 1 <= len(opt) <= 100

    assert metadata[0] == {"article_id": "id_1"}
    assert metadata[1] == {"article_id": "id_2"}
    assert metadata[2] == {"article_id": "id_3"}
    assert metadata[3] == {"article_id": "id_4"}
    assert options[3] == "Untitled item"
    assert metadata[4] == {"none_option": True}


def test_truncate_question() -> None:
    long_q = "How useful was this digest? " * 20
    truncated = truncate_question(long_q)
    assert 1 <= len(truncated) <= 300
    assert truncated.endswith("...")
