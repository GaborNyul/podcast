# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Nightly performance tests (excluded from the default run)."""

import time

import pytest

from podcast.ingest.loader import SourceDocument, merged_sources_markdown
from podcast.script.markdown import markdown_to_transcript, transcript_to_markdown
from podcast.script.models import Transcript, Turn
from podcast.tts.cache import segment_key

pytestmark = pytest.mark.performance


class TestSegmentKeyThroughput:
    def test_five_thousand_lines_under_a_second(self) -> None:
        lines = [f"spoken line number {index} with a bit of text" for index in range(5000)]
        started = time.monotonic()
        keys = {segment_key("kokoro", "af_heart", line) for line in lines}
        elapsed = time.monotonic() - started
        assert len(keys) == 5000
        assert elapsed < 1.0


class TestMarkdownRoundTripThroughput:
    def test_two_thousand_turns_under_two_seconds(self) -> None:
        transcript = Transcript(
            title="Big Episode",
            hosts=["Alex", "Maya"],
            turns=[
                Turn(
                    speaker="Alex" if index % 2 == 0 else "Maya",
                    text=f"turn {index} " + "word " * 30,
                )
                for index in range(2000)
            ],
        )
        started = time.monotonic()
        parsed = markdown_to_transcript(transcript_to_markdown(transcript))
        elapsed = time.monotonic() - started
        assert len(parsed.turns) == 2000
        assert elapsed < 2.0


class TestMergedSourcesThroughput:
    def test_thousand_documents_under_a_second(self) -> None:
        documents = [
            SourceDocument(
                path=f"doc{index}.md",
                title=f"Doc {index}",
                markdown="paragraph " * 200,
                tokens=200,
            )
            for index in range(1000)
        ]
        started = time.monotonic()
        merged = merged_sources_markdown(documents)
        elapsed = time.monotonic() - started
        assert "Source 1000" in merged
        assert elapsed < 1.0
