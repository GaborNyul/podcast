# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.ingest.tokens."""

import math

import pytest

from podcast.errors import IngestError
from podcast.ingest import tokens


class _FakeEncoding:
    """Deterministic encoder double: one token per character."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def _fake_encoder() -> _FakeEncoding:
    return _FakeEncoding()


def _no_encoder() -> None:
    return None


class TestCountTokens:
    def test_uses_encoder_with_safety_factor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", _fake_encoder)
        assert tokens.count_tokens("abcd") == math.ceil(4 * tokens.SAFETY_FACTOR)

    def test_falls_back_to_word_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", _no_encoder)
        expected = math.ceil(math.ceil(3 * 4 / 3) * tokens.SAFETY_FACTOR)
        assert tokens.count_tokens("one two three") == expected

    def test_empty_text_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", _no_encoder)
        assert tokens.count_tokens("") == 0


class TestLoadEncoder:
    def test_offline_load_degrades_to_heuristic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(_name: str) -> None:
            raise RuntimeError("offline")

        tokens.load_encoder.cache_clear()
        monkeypatch.setattr("tiktoken.get_encoding", refuse)
        try:
            assert tokens.load_encoder() is None
            expected = math.ceil(math.ceil(2 * 4 / 3) * tokens.SAFETY_FACTOR)
            assert tokens.count_tokens("two words") == expected
        finally:
            tokens.load_encoder.cache_clear()

    def test_loaded_encoding_drives_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def provide(_name: str) -> _FakeEncoding:
            return _FakeEncoding()

        tokens.load_encoder.cache_clear()
        monkeypatch.setattr("tiktoken.get_encoding", provide)
        try:
            assert tokens.count_tokens("abcd") == math.ceil(4 * tokens.SAFETY_FACTOR)
        finally:
            tokens.load_encoder.cache_clear()


class TestAssertFitsContext:
    def test_fitting_sources_pass(self) -> None:
        tokens.assert_fits_context(1000, context_window=262144)

    def test_exact_budget_passes(self) -> None:
        tokens.assert_fits_context(100, context_window=200, reserve=100)

    def test_overflow_raises_with_numbers(self) -> None:
        with pytest.raises(IngestError, match="262144"):
            tokens.assert_fits_context(300000, context_window=262144)
