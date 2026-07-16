"""Tests for podcast.script.markdown — the lossless round-trip contract."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from podcast import emphasis
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


def _canonical_turn_text(value: str) -> str:
    """The write-side canonical form: emphasis-normalize first, then whitespace-collapse."""
    return normalize_turn_text(emphasis.normalize(value))


# General text plus a star-dense alphabet so emphasis markup (valid and malformed)
# is generated often.
_raw_turn_texts = st.one_of(
    st.text(min_size=0, max_size=200),
    st.text(alphabet=st.sampled_from(list("*ab c!\n")), max_size=40),
)
_turn_text = _raw_turn_texts.map(_canonical_turn_text)
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

    @given(text=_raw_turn_texts)
    def test_round_trip_is_total_for_arbitrary_turn_text(self, text: str) -> None:
        """Serialize-then-parse never raises; the text lands in the canonical form."""
        transcript = Transcript(title="T", hosts=_HOSTS, turns=[Turn(speaker="Alex", text=text)])
        serialized = transcript_to_markdown(transcript)
        parsed = markdown_to_transcript(serialized)
        assert parsed.turns[0].text == _canonical_turn_text(text)
        # Oracle-free idempotence: re-serializing the parse is byte-stable.
        assert transcript_to_markdown(parsed) == serialized

    def test_unicode_title_and_text_survive(self) -> None:
        transcript = Transcript(
            title='Űrhajó: a "nagy" kaland 🚀',
            hosts=_HOSTS,
            turns=[Turn(speaker="Maya", text="Szia! Űrhajó — a „nagy” kaland 🚀 [link] `code`")],
        )
        assert markdown_to_transcript(transcript_to_markdown(transcript)) == transcript

    def test_markdown_bold_is_canonicalized_to_emphasis(self) -> None:
        transcript = Transcript(
            title="T",
            hosts=_HOSTS,
            turns=[Turn(speaker="Maya", text="Szia! **bold** [link] `code`")],
        )
        parsed = markdown_to_transcript(transcript_to_markdown(transcript))
        # `**bold**` is not the emphasis grammar (ADR 0014): write-side
        # canonicalization keeps the inner `*bold*` span and drops the stray
        # outer asterisks; from then on the round-trip is byte-stable.
        assert parsed.turns[0].text == "Szia! *bold* [link] `code`"
        assert markdown_to_transcript(transcript_to_markdown(parsed)) == parsed


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

    def test_stray_asterisks_are_dropped_and_valid_spans_kept(self) -> None:
        line = turn_to_line(Turn(speaker="Alex", text="keep *this*, drop ** and * strays"))
        assert line == "**Alex:** keep *this*, drop and strays"


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

    def test_valid_emphasis_is_preserved_in_turn_text(self) -> None:
        text = (
            '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n'
            "**Alex:** the *whole point*, not a footnote\n"
        )
        assert markdown_to_transcript(text).turns[0].text == "the *whole point*, not a footnote"

    def test_two_char_emphasis_span_survives_the_round_trip(self) -> None:
        # A minimal-width span: two chars exercise the grammar's optional
        # interior with an empty middle; the span must not be dropped as a
        # stray by write-side canonicalization or rejected by the parser.
        transcript = Transcript(
            title="T", hosts=_HOSTS, turns=[Turn(speaker="Alex", text="that's *it* folks")]
        )
        parsed = markdown_to_transcript(transcript_to_markdown(transcript))
        assert parsed.turns[0].text == "that's *it* folks"

    @pytest.mark.parametrize(
        "bad_text",
        ["so **bold** wrong", "a stray * here", "a * padded * span"],
    )
    def test_malformed_emphasis_raises_with_line_number(self, bad_text: str) -> None:
        text = f'---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex:** {bad_text}\n'
        expected = (
            r"line 6 has stray or malformed emphasis"
            r".*stress a word as `\*word\*`"
            r".*literal '\*' cannot be written"
        )
        with pytest.raises(ScriptError, match=expected):
            markdown_to_transcript(text)

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

    def test_closing_bracket_without_opener_is_an_unknown_host(self) -> None:
        text = '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex]:** hi\n'
        with pytest.raises(ScriptError, match="unknown host 'Alex\\]'"):
            markdown_to_transcript(text)

    def test_delivery_note_with_inner_brackets_splits_at_the_first_opener(self) -> None:
        # The hand-edit tolerance the old regex gave: host = text before the
        # FIRST '[', delivery = everything up to the trailing ']' (its own
        # brackets then dropped by normalize_delivery).
        text = (
            '---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**Alex [warm [really] tone]:** hi\n'
        )
        turn = markdown_to_transcript(text).turns[0]
        assert turn.speaker == "Alex"
        assert turn.delivery == "warm really tone"

    def test_pathological_whitespace_speaker_token_fails_fast(self) -> None:
        # The old `^(.*?)\s*\[(.*)\]$` delivery regex backtracked quadratically
        # on a token like this (1.2s CPU at 50k spaces, ~20s at 200k); the
        # string-op split is linear. Plain assertion by design — the suite's
        # normal runtime/timeout is the regression guard.
        token = "Bob" + " " * 50_000 + "x"
        text = f'---\ntitle: "T"\nhosts: ["Alex", "Maya"]\n---\n\n**{token}:** hi\n'
        with pytest.raises(ScriptError, match="unknown host"):
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
