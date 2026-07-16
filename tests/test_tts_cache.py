# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.tts.cache."""

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from podcast.tts import cache

_texts = st.text(max_size=100)


class TestSegmentKey:
    @given(engine=_texts, voice=_texts, text=_texts, delivery=_texts)
    def test_key_is_stable(self, engine: str, voice: str, text: str, delivery: str) -> None:
        assert cache.segment_key(engine, voice, text, delivery) == cache.segment_key(
            engine, voice, text, delivery
        )

    @given(text_a=_texts, text_b=_texts)
    def test_different_text_different_key(self, text_a: str, text_b: str) -> None:
        if text_a != text_b:
            assert cache.segment_key("e", "v", text_a) != cache.segment_key("e", "v", text_b)

    @given(delivery_a=_texts, delivery_b=_texts)
    def test_different_delivery_different_key(self, delivery_a: str, delivery_b: str) -> None:
        if delivery_a != delivery_b:
            assert cache.segment_key("e", "v", "t", delivery_a) != cache.segment_key(
                "e", "v", "t", delivery_b
            )

    def test_field_boundaries_do_not_collide(self) -> None:
        # ("ab", "c") vs ("a", "bc") must hash differently despite equal concatenation.
        assert cache.segment_key("e", "ab", "c") != cache.segment_key("e", "a", "bc")
        assert cache.segment_key("e", "v", "ab", "c") != cache.segment_key("e", "v", "a", "bc")

    def test_key_is_hex_sha256(self) -> None:
        key = cache.segment_key("kokoro", "af_heart", "hello")
        assert len(key) == 64
        assert all(character in "0123456789abcdef" for character in key)


class TestEnsureSegment:
    def test_miss_renders_then_hit_skips(self, tmp_path: Path) -> None:
        stats = cache.CacheStats()
        renders: list[Path] = []

        def render(path: Path) -> None:
            renders.append(path)
            path.write_bytes(b"RIFFfake")

        first = cache.ensure_segment(tmp_path / "seg", "e", "v", "hi", "", render, stats)
        second = cache.ensure_segment(tmp_path / "seg", "e", "v", "hi", "", render, stats)
        assert first == second
        assert len(renders) == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total == 2

    def test_render_writes_to_scratch_then_moves(self, tmp_path: Path) -> None:
        stats = cache.CacheStats()

        def render(path: Path) -> None:
            assert path.suffix == ".wav"
            assert ".tmp" in path.name
            path.write_bytes(b"RIFF")

        final = cache.ensure_segment(tmp_path, "e", "v", "hi", "", render, stats)
        assert final.is_file()
        assert not list(tmp_path.glob("*.tmp.wav"))

    def test_edited_line_changes_only_its_segment(self, tmp_path: Path) -> None:
        stats = cache.CacheStats()

        def render(path: Path) -> None:
            path.write_bytes(b"RIFF")

        original = cache.ensure_segment(tmp_path, "e", "v", "line one", "", render, stats)
        edited = cache.ensure_segment(tmp_path, "e", "v", "line one, edited", "", render, stats)
        assert original != edited
        assert original.is_file()

    def test_edited_delivery_note_re_renders_the_line(self, tmp_path: Path) -> None:
        stats = cache.CacheStats()

        def render(path: Path) -> None:
            path.write_bytes(b"RIFF")

        neutral = cache.ensure_segment(tmp_path, "e", "v", "line", "", render, stats)
        excited = cache.ensure_segment(tmp_path, "e", "v", "line", "excited", render, stats)
        assert neutral != excited
        assert stats.misses == 2
