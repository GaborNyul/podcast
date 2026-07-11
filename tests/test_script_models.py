"""Tests for podcast.script.models."""

import pytest
from pydantic import ValidationError

from podcast.script.models import Outline, OutlineSegment, Transcript, Turn


class TestTranscript:
    def test_word_count_sums_turns(self) -> None:
        transcript = Transcript(
            title="T",
            hosts=["A", "B"],
            turns=[Turn(speaker="A", text="one two"), Turn(speaker="B", text="three")],
        )
        assert transcript.word_count() == 3

    def test_empty_turns_count_zero(self) -> None:
        transcript = Transcript(title="T", hosts=["A", "B"], turns=[])
        assert transcript.word_count() == 0

    def test_requires_two_hosts(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(title="T", hosts=["Solo"], turns=[])

    def test_extra_llm_keys_are_ignored(self) -> None:
        transcript = Transcript.model_validate(
            {"title": "T", "hosts": ["A", "B"], "turns": [], "confidence": 0.9}
        )
        assert not hasattr(transcript, "confidence")

    def test_turn_delivery_defaults_empty_and_never_counts_as_words(self) -> None:
        turn = Turn(speaker="A", text="one two", delivery="excited, racing ahead")
        assert Turn(speaker="A", text="x").delivery == ""
        transcript = Transcript(title="T", hosts=["A", "B"], turns=[turn])
        assert transcript.word_count() == 2


class TestOutline:
    def test_total_words(self) -> None:
        outline = Outline(
            title="T",
            segments=[
                OutlineSegment(heading="a", target_words=100),
                OutlineSegment(heading="b", target_words=50),
            ],
        )
        assert outline.total_words() == 150

    def test_requires_a_segment(self) -> None:
        with pytest.raises(ValidationError):
            Outline(title="T", segments=[])

    def test_rejects_zero_word_segment(self) -> None:
        with pytest.raises(ValidationError):
            OutlineSegment(heading="a", target_words=0)

    def test_notes_default_empty(self) -> None:
        assert OutlineSegment(heading="a", target_words=1).notes == ""
