"""Tests for podcast.script.formats (registry, prompt composition, invariants)."""

import hashlib

import pytest

from podcast.errors import ConfigError
from podcast.script import formats, prompts

# The validated hybrid-v2 deep-dive prompt (ADR 0009). If this pin fails, the
# prompt's bytes changed — that must be a deliberate, bakeoff-backed decision.
_DEEP_DIVE_SHA256 = (
    "90040487f5c247f3b5ac7f4fcbbcd3ad"  # pragma: allowlist secret — prompt hash pin
    "66734882659bdc0a21c21c1705d25220"  # pragma: allowlist secret
)


class TestRegistry:
    def test_ships_all_four_formats(self) -> None:
        assert list(formats.FORMATS) == ["deep-dive", "brief", "debate", "critique"]

    def test_resolve_returns_the_spec(self) -> None:
        assert formats.resolve("brief").key == "brief"

    def test_unknown_key_fails_listing_choices(self) -> None:
        with pytest.raises(ConfigError, match="deep-dive, brief, debate, critique"):
            formats.resolve("sirens")

    @pytest.mark.parametrize("key", list(formats.FORMATS))
    def test_specs_are_shaped_sanely(self, key: str) -> None:
        spec = formats.FORMATS[key]
        low, high = spec.segment_range
        assert 1 <= low <= high
        assert spec.speakers in (1, 2)
        assert spec.description
        assert spec.label
        if spec.default_minutes is not None:
            assert spec.default_minutes >= 1


class TestDeepDiveIdentity:
    def test_deep_dive_is_the_validated_prompt_object(self) -> None:
        spec = formats.FORMATS["deep-dive"]
        assert spec.system_prompt is prompts.SYSTEM_PROMPT
        assert spec.outline_brief is prompts.OUTLINE_BRIEF
        assert spec.polish_brief is prompts.POLISH_BRIEF
        assert spec.opening_position is prompts.OPENING_POSITION
        assert spec.continuing_position is prompts.CONTINUING_POSITION
        assert spec.final_position is prompts.FINAL_POSITION

    def test_hybrid_v2_bytes_are_pinned(self) -> None:
        digest = hashlib.sha256(prompts.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        assert digest == _DEEP_DIVE_SHA256, (
            "the deep-dive system prompt changed; it is the bakeoff-validated "
            "hybrid-v2 text (ADR 0009) — change it deliberately and update the pin"
        )

    def test_deep_dive_defaults_defer_to_config(self) -> None:
        assert formats.FORMATS["deep-dive"].default_minutes is None


class TestSharedInvariants:
    """The invariant core every format must carry (TTS and grounding rules)."""

    @pytest.mark.parametrize("key", list(formats.FORMATS))
    def test_audio_rules_are_shared_verbatim(self, key: str) -> None:
        prompt = formats.FORMATS[key].system_prompt
        assert formats.AUDIO_BLOCK in prompt  # bans [laughs]-style cues, defines delivery
        assert "no bracketed cues like [laughs]" in prompt

    @pytest.mark.parametrize("key", list(formats.FORMATS))
    def test_listener_and_grounding_are_shared(self, key: str) -> None:
        prompt = formats.FORMATS[key].system_prompt
        assert formats.LISTENER_BLOCK in prompt
        assert "Everything asserted comes from the provided sources" in prompt
        assert "Attribution hygiene" in prompt

    def test_neutrality_holds_for_summary_formats_only(self) -> None:
        assert formats.NEUTRAL_STANCE_RULE in formats.FORMATS["deep-dive"].system_prompt
        assert formats.NEUTRAL_STANCE_RULE in formats.FORMATS["brief"].system_prompt
        assert formats.NEUTRAL_STANCE_RULE not in formats.FORMATS["debate"].system_prompt
        assert formats.NEUTRAL_STANCE_RULE not in formats.FORMATS["critique"].system_prompt

    def test_grounding_splice_guard_fires_when_sentence_vanishes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(formats, "_GROUNDING_BLOCK", "GROUNDING\nSomething else.")
        with pytest.raises(AssertionError, match="neutrality sentence"):
            formats.grounding_block("any stance")


class TestBrief:
    def test_is_solo_with_a_hard_guard(self) -> None:
        spec = formats.FORMATS["brief"]
        assert spec.speakers == 1
        assert "SOLO episode with only ONE speaker" in spec.system_prompt
        assert "Do NOT invent or add any other speakers" in spec.system_prompt
        assert "SOLO script" in spec.polish_brief

    def test_short_is_accepted_never_padded(self) -> None:
        spec = formats.FORMATS["brief"]
        assert spec.length_mode == "ceiling"
        assert spec.segment_range == (1, 2)
        assert spec.default_minutes == 2

    def test_bans_generic_intros(self) -> None:
        assert "Never a generic welcome" in formats.FORMATS["brief"].system_prompt


class TestDebate:
    def test_assigns_stances_and_never_pads(self) -> None:
        spec = formats.FORMATS["debate"]
        assert spec.assigns_stances
        assert "argument thread" in spec.extend_guidance
        assert "host_angles" in spec.outline_brief

    def test_adversarial_register_rules(self) -> None:
        spec = formats.FORMATS["debate"]
        assert "Agreement glue is OFF" in spec.system_prompt
        assert "never switching sides" in spec.system_prompt
        assert "Steelman" in spec.system_prompt
        assert "PRESERVING the adversarial register" in spec.polish_brief

    def test_ends_without_a_verdict(self) -> None:
        spec = formats.FORMATS["debate"]
        assert "no verdict" in spec.system_prompt
        assert "no verdict" in spec.final_position


class TestCritique:
    def test_has_a_review_stage_with_anchor_rules(self) -> None:
        spec = formats.FORMATS["critique"]
        assert spec.review_prompt
        assert "anchors to something the document actually says" in spec.review_prompt
        assert "better to say less than to make things up" in spec.review_prompt

    def test_overrides_neutrality_explicitly(self) -> None:
        spec = formats.FORMATS["critique"]
        assert "takes positions about the material's quality" in spec.system_prompt
        assert "No compliment sandwich" in spec.system_prompt

    def test_expansion_adds_findings_not_padding(self) -> None:
        assert "anchored findings" in formats.FORMATS["critique"].extend_guidance
