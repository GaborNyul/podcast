# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.emphasis — the `*word*` stress-markup grammar (ADR 0014)."""

import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from podcast.emphasis import (
    EMPHASIS_RE,
    normalize,
    render,
    render_caps,
    spans,
    strip_markup,
    validate,
)

# General text plus a star-dense alphabet so malformed markup is generated often.
_texts = st.one_of(
    st.text(max_size=80),
    st.text(alphabet=st.sampled_from(list("*ab c!\n")), max_size=40),
)


class TestEmphasisRe:
    def test_group_one_is_the_span_text(self) -> None:
        match = EMPHASIS_RE.search("say *two words* now")
        assert match is not None
        assert match.group(1) == "two words"

    @pytest.mark.parametrize(
        "text", ["**", "* *", "* padded*", "*padded *", "*", "*a", "*foo\nbar*"]
    )
    def test_malformed_markup_never_matches(self, text: str) -> None:
        assert EMPHASIS_RE.search(text) is None


class TestSpans:
    def test_spans_in_document_order(self) -> None:
        assert spans("*a* then *b* and *c*") == ["a", "b", "c"]

    def test_no_markup_yields_empty_list(self) -> None:
        assert spans("plain text") == []

    def test_empty_string_yields_empty_list(self) -> None:
        assert spans("") == []

    def test_multi_word_span(self) -> None:
        assert spans("the *whole point* is") == ["whole point"]

    def test_two_char_span_matches(self) -> None:
        # Exactly two chars exercises the grammar's optional interior: first
        # char + empty middle + last char ('*i*' and '*it.*' cannot catch a
        # middle that demands one-or-more characters, but '*it*' does).
        assert spans("that's *it* folks") == ["it"]

    def test_bold_yields_only_the_inner_span(self) -> None:
        assert spans("**bold**") == ["bold"]

    def test_span_adjacent_to_punctuation(self) -> None:
        assert spans("wait (*really*)?!") == ["really"]

    def test_spans_at_start_and_end_of_string(self) -> None:
        assert spans("*Start* middle *end*") == ["Start", "end"]

    def test_span_never_crosses_a_newline(self) -> None:
        assert spans("say *foo\nbar* now") == []

    def test_adjacent_spans_both_match(self) -> None:
        assert spans("*a**b*") == ["a", "b"]


class TestStripMarkup:
    def test_unwraps_valid_span(self) -> None:
        assert strip_markup("That's the *whole* point") == "That's the whole point"

    def test_drops_stray_asterisk_without_touching_whitespace(self) -> None:
        assert strip_markup("a * b") == "a  b"

    def test_output_never_contains_asterisk(self) -> None:
        assert "*" not in strip_markup("**bold** *x* y* * z *")

    def test_empty_string(self) -> None:
        assert strip_markup("") == ""

    def test_whitespace_is_preserved(self) -> None:
        assert strip_markup("  *a*   b\t ") == "  a   b\t "


class TestNormalize:
    def test_valid_spans_are_kept_intact(self) -> None:
        assert normalize("keep *this* and *that one* here") == "keep *this* and *that one* here"

    def test_unpaired_asterisk_is_dropped(self) -> None:
        assert normalize("keep *this here") == "keep this here"

    def test_bold_normalizes_to_single_emphasis(self) -> None:
        # Documented behavior: the inner span is valid, the outer asterisks are strays.
        assert normalize("**bold**") == "*bold*"

    def test_empty_span_is_dropped(self) -> None:
        assert normalize("**") == ""

    def test_padded_span_loses_asterisks_keeps_text(self) -> None:
        assert normalize("* padded *") == " padded "

    def test_stray_after_valid_span_is_dropped(self) -> None:
        assert normalize("*a*b*") == "*a*b"

    def test_whitespace_is_never_collapsed_or_trimmed(self) -> None:
        assert normalize("  a \t b\n") == "  a \t b\n"

    def test_spans_at_start_and_end_of_string(self) -> None:
        assert normalize("*Start* and *end*") == "*Start* and *end*"

    def test_asterisks_paired_across_a_newline_are_strays(self) -> None:
        assert normalize("say *foo\nbar* now") == "say foo\nbar now"


class TestValidate:
    @pytest.mark.parametrize(
        "text",
        ["", "no markup at all", "say *two words* loud", "*a* mid *b*", "*a**b*", "*it*"],
    )
    def test_accepts_clean_and_valid_text(self, text: str) -> None:
        validate(text)

    @pytest.mark.parametrize(
        "text", ["*word", "**", "* padded *", "**bold**", "*a* * *b*", "*foo\nbar*"]
    )
    def test_raises_on_malformed_markup(self, text: str) -> None:
        with pytest.raises(ValueError, match="emphasis"):
            validate(text)

    def test_error_names_the_offending_fragment(self) -> None:
        with pytest.raises(ValueError, match=re.escape("stray * here")):
            validate("ok *fine* but stray * here")


class TestRender:
    def test_render_span_is_applied_to_each_valid_span(self) -> None:
        # The shape the SoulX renderer will use (ADR 0014).
        marked = render("a *b* and *c d*", lambda span: f"<|s|>{span}<|e|>")
        assert marked == "a <|s|>b<|e|> and <|s|>c d<|e|>"

    def test_strays_are_dropped_and_plain_text_kept(self) -> None:
        assert render("a * b *c*", str.upper) == "a  b C"

    def test_whitespace_is_untouched(self) -> None:
        assert render("  no markup\t here ", str.upper) == "  no markup\t here "


class TestRenderCaps:
    def test_span_becomes_caps(self) -> None:
        assert render_caps("That's the *whole* point") == "That's the WHOLE point"

    def test_multi_word_span_becomes_caps(self) -> None:
        assert render_caps("*two words*") == "TWO WORDS"

    def test_strays_are_dropped(self) -> None:
        assert render_caps("a * b *c*") == "a  b C"

    def test_bold_renders_inner_span(self) -> None:
        assert render_caps("**bold**") == "BOLD"

    def test_text_without_markup_is_unchanged(self) -> None:
        assert render_caps("plain text") == "plain text"

    def test_length_changing_unicode_uppercase(self) -> None:
        # qwen3's instruct clause names the original span while the text renders longer.
        assert render_caps("*straße*") == "STRASSE"


class TestProperties:
    @given(text=_texts)
    def test_strip_markup_is_idempotent(self, text: str) -> None:
        assert strip_markup(strip_markup(text)) == strip_markup(text)

    @given(text=_texts)
    def test_normalize_is_idempotent(self, text: str) -> None:
        assert normalize(normalize(text)) == normalize(text)

    @given(text=_texts)
    def test_validate_never_raises_on_normalized_text(self, text: str) -> None:
        validate(normalize(text))

    @given(text=_texts)
    def test_strip_of_normalized_equals_strip(self, text: str) -> None:
        assert strip_markup(normalize(text)) == strip_markup(text)

    @given(text=_texts)
    def test_strip_output_never_contains_asterisk(self, text: str) -> None:
        assert "*" not in strip_markup(text)

    @given(text=_texts)
    def test_normalize_never_loses_or_invents_spans(self, text: str) -> None:
        assert spans(normalize(text)) == spans(text)

    @given(text=_texts)
    def test_validate_raises_exactly_when_normalize_would_change_text(self, text: str) -> None:
        if normalize(text) != text:
            with pytest.raises(ValueError, match="emphasis"):
                validate(text)
        else:
            validate(text)
