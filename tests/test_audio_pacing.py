"""Tests for podcast.audio.pacing (content-aware inter-turn pause scales)."""

from podcast.audio import pacing
from podcast.script.models import Turn


def _turn(text: str) -> Turn:
    return Turn(speaker="Alex", text=text)


LONG_REPLY = "That is a genuinely longer reply that keeps the conversation moving forward."


class TestGapScales:
    def test_no_turns_no_gaps(self) -> None:
        assert pacing.gap_scales([]) == []
        assert pacing.gap_scales([_turn("Hi.")]) == []

    def test_default_scale_for_plain_statements(self) -> None:
        assert pacing.gap_scales([_turn("We start here."), _turn(LONG_REPLY)]) == [1.0]

    def test_interruption_dash_gets_shortest_gap(self) -> None:
        assert pacing.gap_scales([_turn("And then it just—"), _turn(LONG_REPLY)]) == [
            pacing.INTERRUPT_SCALE
        ]

    def test_backchannel_lands_fast(self) -> None:
        assert pacing.gap_scales([_turn(LONG_REPLY), _turn("Right, exactly.")]) == [
            pacing.BACKCHANNEL_SCALE
        ]

    def test_interruption_beats_backchannel(self) -> None:
        assert pacing.gap_scales([_turn("so if we—"), _turn("Wait.")]) == [pacing.INTERRUPT_SCALE]

    def test_question_handoff_stays_snappy(self) -> None:
        assert pacing.gap_scales([_turn("But who pays for that?"), _turn(LONG_REPLY)]) == [
            pacing.HANDOFF_SCALE
        ]

    def test_ellipsis_earns_a_beat(self) -> None:
        assert pacing.gap_scales([_turn("Okay, get this..."), _turn(LONG_REPLY)]) == [
            pacing.BEAT_SCALE
        ]

    def test_unicode_ellipsis_earns_a_beat(self) -> None:
        assert pacing.gap_scales([_turn("Okay, get this…"), _turn(LONG_REPLY)]) == [
            pacing.BEAT_SCALE
        ]

    def test_one_scale_per_gap(self) -> None:
        turns = [_turn("First?"), _turn(LONG_REPLY), _turn("Totally.")]
        assert pacing.gap_scales(turns) == [pacing.HANDOFF_SCALE, pacing.BACKCHANNEL_SCALE]
