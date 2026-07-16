# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.script.markdown — the lossless round-trip contract."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from podcast.errors import ScriptError
from podcast.script.markdown import (
    markdown_to_transcript,
    normalize_delivery,
    normalize_turn_text,
    transcript_to_markdown,
    turn_to_line,
)
from podcast.script.models import Transcript, Turn

_HOSTS = ["Alex", "Maya"]

_turn_text = st.text(min_size=0, max_size=200).map(normalize_turn_text)
_deliveries = st.text(min_size=0, max_size=60).map(normalize_delivery)
_titles = st.text(min_size=1, max_size=80).filter(lambda value: value.strip() != "")


def _transcripts() -> st.SearchStrategy[Transcript]:
    turns = st.lists(
        st.builds(
            Turn,
            speaker=st.sampled_from(_HOSTS),
            text=_turn_text,
            delivery=_deliveries,
        ),
        max_size=20,
    )
    return st.builds(Transcript, title=_titles, hosts=st.just(list(_HOSTS)), turns=turns)


class TestRoundTrip:
    @given(transcript=_transcripts())
    def test_parse_inverts_serialize(self, transcript: Transcript) -> None:
        assert markdown_to_transcript(transcript_to_markdown(transcript)) == transcript

    def test_normalization_makes_serialize_idempotent(self) -> None:
        messy = Transcript(
            title="T",
            hosts=_HOSTS,
            turns=[Turn(speaker="Alex", text="  line\none\t two ")],
        )
        once = transcript_to_markdown(messy)
        again = transcript_to_markdown(markdown_to_transcript(once))
        assert once == again

    def test_unicode_title_and_text_survive(self) -> None:
        transcript = Transcript(
            title='Űrhajó: a "nagy" kaland 🚀',
            hosts=_HOSTS,
            turns=[Turn(speaker="Maya", text="Szia! **bold** [link] `code`")],
        )
        assert markdown_to_transcript(transcript_to_markdown(transcript)) == transcript


class TestTranscriptToMarkdown:
    def test_shape_of_output(self) -> None:
        transcript = Transcript(
            title="Ants", hosts=_HOSTS, turns=[Turn(speaker="Alex", text="Hi.")]
        )
        text = transcript_to_markdown(transcript)
        assert text.startswith(
            '---\ntitle: "Ants"\nhosts: ["Alex", "Maya"]\nformat: "deep-dive"\n---\n'
        )
        assert "**Alex:** Hi." in text
        assert "Edit freely" in text

    def test_format_line_round_trips(self) -> None:
        transcript = Transcript(
            title="T", hosts=["Maya"], turns=[Turn(speaker="Maya", text="Hi.")], format="brief"
        )
        text = transcript_to_markdown(transcript)
        assert 'format: "brief"' in text
        parsed = markdown_to_transcript(text)
        assert parsed == transcript
        assert parsed.format == "brief"
        assert parsed.hosts == ["Maya"]

    def test_missing_format_line_defaults_to_deep_dive(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex:** hi\n'
        assert markdown_to_transcript(text).format == "deep-dive"

    def test_delivery_note_rides_in_the_speaker_token(self) -> None:
        transcript = Transcript(
            title="T",
            hosts=_HOSTS,
            turns=[Turn(speaker="Maya", text="Get this.", delivery="excited, leaning in")],
        )
        assert "**Maya [excited, leaning in]:** Get this." in transcript_to_markdown(transcript)


class TestTurnToLine:
    def test_empty_delivery_keeps_plain_format(self) -> None:
        assert turn_to_line(Turn(speaker="Alex", text="Hi.")) == "**Alex:** Hi."

    def test_grammar_breaking_characters_are_dropped_from_delivery(self) -> None:
        line = turn_to_line(Turn(speaker="Alex", text="Hi.", delivery="wry: [aside] beat"))
        assert line == "**Alex [wry aside beat]:** Hi."


class TestMarkdownToTranscript:
    def test_missing_front_matter_raises(self) -> None:
        with pytest.raises(ScriptError, match="front-matter"):
            markdown_to_transcript("**Alex:** hi")

    def test_unterminated_front_matter_raises(self) -> None:
        with pytest.raises(ScriptError, match="unterminated"):
            markdown_to_transcript('---\ntitle: "T"\nhosts: ["A", "B"]\n')

    def test_missing_hosts_raises(self) -> None:
        with pytest.raises(ScriptError, match="title and hosts"):
            markdown_to_transcript('---\ntitle: "T"\n---\n')

    def test_non_json_front_matter_value_raises(self) -> None:
        with pytest.raises(ScriptError, match="not JSON"):
            markdown_to_transcript("---\ntitle: unquoted\n---\n")

    def test_bad_body_line_raises_with_line_number(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\njust prose\n'
        with pytest.raises(ScriptError, match="line 6"):
            markdown_to_transcript(text)

    def test_unknown_speaker_raises(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Zed:** hello\n'
        with pytest.raises(ScriptError, match="unknown host 'Zed'"):
            markdown_to_transcript(text)

    def test_front_matter_key_order_is_free(self) -> None:
        text = '---\nhosts: ["Alex", "Maya"]\ntitle: "T"\n---\n\n**Alex:** hi\n'
        transcript = markdown_to_transcript(text)
        assert transcript.title == "T"

    def test_unknown_front_matter_keys_are_ignored(self) -> None:
        text = '---\ntitle: "T"\ndate: "2026-07-10"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex:** hi\n'
        assert markdown_to_transcript(text).title == "T"

    def test_comments_and_blanks_are_skipped(self) -> None:
        text = (
            '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n'
            "<!-- a comment -->\n\n**Maya:** hi there\n"
        )
        transcript = markdown_to_transcript(text)
        assert transcript.turns == [Turn(speaker="Maya", text="hi there")]

    def test_empty_turn_text_is_allowed(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex:**\n'
        assert markdown_to_transcript(text).turns[0].text == ""

    def test_hand_edited_line_parses(self) -> None:
        text = (
            '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n'
            "**Alex:** I rewrote this line by hand!\n"
        )
        transcript = markdown_to_transcript(text)
        assert transcript.turns[0].text == "I rewrote this line by hand!"

    def test_delivery_note_is_parsed_from_speaker_token(self) -> None:
        text = (
            '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n'
            "**Alex [skeptical,  slowing down]:** Hold on.\n"
        )
        turn = markdown_to_transcript(text).turns[0]
        assert turn.speaker == "Alex"
        assert turn.delivery == "skeptical, slowing down"
        assert turn.text == "Hold on."

    def test_empty_brackets_mean_no_delivery(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex []:** hi\n'
        turn = markdown_to_transcript(text).turns[0]
        assert turn.speaker == "Alex"
        assert turn.delivery == ""

    def test_unknown_host_with_delivery_raises(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Zed [warm]:** hello\n'
        with pytest.raises(ScriptError, match="unknown host 'Zed \\[warm\\]'"):
            markdown_to_transcript(text)

    def test_host_name_with_grammar_characters_is_rejected(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex [AI]", "Maya"]\n---\n\n**Maya:** hi\n'
        with pytest.raises(ScriptError, match="may not contain"):
            markdown_to_transcript(text)

    def test_host_name_with_colon_is_rejected(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["DJ: Live", "Maya"]\n---\n\n**Maya:** hi\n'
        with pytest.raises(ScriptError, match="may not contain"):
            markdown_to_transcript(text)


class TestNormalizeTurnText:
    def test_collapses_whitespace(self) -> None:
        assert normalize_turn_text(" a\n b\t\tc ") == "a b c"

    def test_empty_stays_empty(self) -> None:
        assert normalize_turn_text("  \n ") == ""


class TestNormalizeDelivery:
    def test_collapses_whitespace(self) -> None:
        assert normalize_delivery(" warm,\n curious ") == "warm, curious"

    def test_drops_grammar_breaking_characters(self) -> None:
        assert normalize_delivery("a[b]c:d") == "a b c d"

    def test_is_idempotent(self) -> None:
        once = normalize_delivery("odd: [note]")
        assert normalize_delivery(once) == once
