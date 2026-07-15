"""Tests for podcast.script.pipeline (fake + scripted providers)."""

import json
from collections.abc import Mapping, Sequence

import pytest

from podcast.config import AppConfig, HostSpec
from podcast.errors import ScriptError
from podcast.llm.base import ChatMessage
from podcast.llm.fake import FakeProvider
from podcast.script import formats, pipeline, prompts
from podcast.script.models import Outline, OutlineSegment, Transcript, Turn

SOURCES = "## Source 1: Ants\n\nAnts are fascinating."


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.schemas: list[Mapping[str, object] | None] = []
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        del temperature
        self.schemas.append(json_schema)
        self.prompts.append(messages[-2].content if len(messages) >= 2 else "")
        self.systems.append(messages[0].content if messages[0].role == "system" else "")
        return self.replies.pop(0)


class TestBuildOutline:
    def test_fake_provider_outline_sums_to_budget(self) -> None:
        config = AppConfig()
        outline = pipeline.build_outline(FakeProvider(), config, SOURCES, 450)
        assert outline.total_words() == 450
        assert outline.segments

    def test_off_budget_outline_is_rescaled(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [
                    {"heading": "a", "target_words": 100},
                    {"heading": "b", "target_words": 100},
                ],
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, AppConfig(), SOURCES, 300)
        assert outline.total_words() == 300
        assert [segment.target_words for segment in outline.segments] == [150, 150]

    def test_rounding_remainder_lands_on_last_segment(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [
                    {"heading": "a", "target_words": 1},
                    {"heading": "b", "target_words": 1},
                    {"heading": "c", "target_words": 1},
                ],
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, AppConfig(), SOURCES, 100)
        assert outline.total_words() == 100

    def test_prompt_carries_deep_dive_system_arc_and_coverage(self) -> None:
        reply = json.dumps({"title": "T", "segments": [{"heading": "a", "target_words": 50}]})
        provider = _ScriptedProvider([reply])
        pipeline.build_outline(provider, AppConfig(), SOURCES, 50)
        assert provider.systems[0] == prompts.SYSTEM_PROMPT
        assert prompts.OUTLINE_BRIEF in provider.prompts[0]
        assert "approximately 50 words" in provider.prompts[0]


class TestWriteDialogue:
    def test_fake_provider_generates_full_transcript(self) -> None:
        config = AppConfig()
        outline = pipeline.build_outline(FakeProvider(), config, SOURCES, 300)
        seen: list[int] = []
        transcript = pipeline.write_dialogue(FakeProvider(), config, SOURCES, outline, seen.append)
        assert transcript.hosts == ["Alex", "Maya"]
        assert transcript.word_count() == 300
        assert seen == list(range(len(outline.segments)))
        assert {turn.speaker for turn in transcript.turns} == {"Alex", "Maya"}

    def test_speaker_enum_injected_into_schema(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": "hello"}]})
        provider = _ScriptedProvider([reply])
        pipeline.write_dialogue(provider, config, SOURCES, outline)
        schema = provider.schemas[0]
        assert schema is not None
        rendered = json.dumps(schema)
        assert '"enum": ["Alex", "Maya"]' in rendered

    def test_case_insensitive_speaker_is_normalized(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "maya ", "text": "hi"}]})
        provider = _ScriptedProvider([reply])
        transcript = pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert transcript.turns[0].speaker == "Maya"

    def test_unknown_speaker_raises_script_error(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Narrator", "text": "hi"}]})
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match="unknown speaker 'Narrator'"):
            pipeline.write_dialogue(provider, config, SOURCES, outline)

    def test_speaker_echoing_quoted_note_notation_is_resolved(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "alex [warm, curious]", "text": "hi"}]})
        provider = _ScriptedProvider([reply])
        transcript = pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert transcript.turns[0].speaker == "Alex"

    def test_unknown_speaker_with_note_suffix_still_raises(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Narrator [warm]", "text": "hi"}]})
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match=r"unknown speaker 'Narrator \[warm\]'"):
            pipeline.write_dialogue(provider, config, SOURCES, outline)

    def test_empty_segment_raises_script_error(self) -> None:
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": "   "}]})
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match="produced no dialogue"):
            pipeline.write_dialogue(provider, config, SOURCES, outline)

    def test_later_segments_carry_context_and_wrap_up(self) -> None:
        config = AppConfig()
        outline = Outline(
            title="T",
            segments=[
                OutlineSegment(heading="one", target_words=60),
                OutlineSegment(heading="two", target_words=60),
            ],
        )
        turn = {"speaker": "Alex", "text": "context line"}
        provider = _ScriptedProvider([json.dumps({"turns": [turn]}), json.dumps({"turns": [turn]})])
        pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert "warm welcome" in provider.prompts[0]
        assert "context line" in provider.prompts[1]
        assert "wrap up" in provider.prompts[1]

    def test_delivery_notes_survive_and_reach_segment_context(self) -> None:
        config = AppConfig()
        outline = Outline(
            title="T",
            segments=[
                OutlineSegment(heading="one", target_words=60),
                OutlineSegment(heading="two", target_words=60),
            ],
        )
        noted = {"speaker": "Alex", "text": "Get this.", "delivery": "excited, leaning in"}
        plain = {"speaker": "Maya", "text": "Go on."}
        provider = _ScriptedProvider(
            [json.dumps({"turns": [noted]}), json.dumps({"turns": [plain]})]
        )
        transcript = pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert transcript.turns[0].delivery == "excited, leaning in"
        assert transcript.turns[1].delivery == ""
        assert "**Alex [excited, leaning in]:** Get this." in provider.prompts[1]

    def test_malformed_emphasis_is_normalized(self) -> None:
        """The dialogue funnel emphasis-normalizes every LLM reply (ADR 0014)."""
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        text = "this is **bold** and *valid* and stray* stuff"
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": text}]})
        provider = _ScriptedProvider([reply])
        transcript = pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert transcript.turns[0].text == "this is *bold* and *valid* and stray stuff"

    def test_asterisk_only_turn_is_dropped(self) -> None:
        """A turn that normalizes to blank is filtered out, not kept as noise."""
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        turns = [
            {"speaker": "Alex", "text": " ** "},
            {"speaker": "Maya", "text": "a real line"},
        ]
        provider = _ScriptedProvider([json.dumps({"turns": turns})])
        transcript = pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert [turn.text for turn in transcript.turns] == ["a real line"]

    def test_segment_of_only_asterisk_turns_counts_as_empty(self) -> None:
        """Emptiness is judged on the normalized text, so `*` alone is no dialogue."""
        config = AppConfig()
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": "*"}]})
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match="produced no dialogue"):
            pipeline.write_dialogue(provider, config, SOURCES, outline)

    def test_position_hints_carry_deep_dive_rituals(self) -> None:
        config = AppConfig()
        outline = Outline(
            title="T",
            segments=[
                OutlineSegment(heading="one", target_words=60),
                OutlineSegment(heading="two", target_words=60),
                OutlineSegment(heading="three", target_words=60),
            ],
        )
        turn = {"speaker": "Alex", "text": "line"}
        reply = json.dumps({"turns": [turn]})
        provider = _ScriptedProvider([reply, reply, reply])
        pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert provider.systems[0] == prompts.SYSTEM_PROMPT
        assert prompts.OPENING_POSITION in provider.prompts[0]
        assert prompts.FINAL_POSITION not in provider.prompts[0]
        assert prompts.CONTINUING_POSITION in provider.prompts[1]
        assert prompts.FINAL_POSITION not in provider.prompts[1]
        assert prompts.CONTINUING_POSITION + prompts.FINAL_POSITION in provider.prompts[2]


class TestPolishDialogue:
    def _transcript(self) -> Transcript:
        return Transcript(
            title="T",
            hosts=["Alex", "Maya"],
            turns=[
                Turn(speaker="Alex", text="one two three four five", delivery="wry"),
                Turn(speaker="Maya", text="six seven eight nine ten"),
            ],
        )

    def test_polish_prompt_carries_brief_script_and_word_target(self) -> None:
        polished = [{"speaker": "Maya", "text": "better radio line", "delivery": "amused"}]
        provider = _ScriptedProvider([json.dumps({"turns": polished})])
        result = pipeline.polish_dialogue(provider, AppConfig(), self._transcript())
        assert provider.systems[0] == prompts.SYSTEM_PROMPT
        assert prompts.POLISH_BRIEF in provider.prompts[0]
        assert "approximately 10 words" in provider.prompts[0]
        assert "**Alex [wry]:** one two three four five" in provider.prompts[0]
        assert result.turns[0].delivery == "amused"

    def test_polish_reply_emphasis_is_normalized(self) -> None:
        """Polish rides the same funnel, so its replies are normalized too."""
        polished = [{"speaker": "Maya", "text": "keep *this* but not **that"}]
        provider = _ScriptedProvider([json.dumps({"turns": polished})])
        result = pipeline.polish_dialogue(provider, AppConfig(), self._transcript())
        assert result.turns[0].text == "keep *this* but not that"

    def test_disabled_polish_pass_skips_the_llm(self) -> None:
        config = AppConfig()
        config.script.polish_pass = False
        provider = _ScriptedProvider([])
        transcript = self._transcript()
        assert pipeline.polish_dialogue(provider, config, transcript) is transcript
        assert provider.schemas == []

    def test_empty_polish_reply_keeps_the_original(self) -> None:
        provider = _ScriptedProvider([json.dumps({"turns": []})])
        transcript = self._transcript()
        assert pipeline.polish_dialogue(provider, AppConfig(), transcript) is transcript


class TestEnsureLength:
    def _transcript(self, words: int) -> Transcript:
        return Transcript(
            title="T", hosts=["Alex", "Maya"], turns=[Turn(speaker="Alex", text="w " * words)]
        )

    def test_within_tolerance_returns_unchanged(self) -> None:
        transcript = self._transcript(100)
        provider = _ScriptedProvider([])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert result is transcript
        assert provider.schemas == []

    def test_short_script_is_expanded(self) -> None:
        transcript = self._transcript(50)
        repaired_turns = [{"speaker": "Maya", "text": "word " * 100}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert result.word_count() == 100
        assert "Expand" in provider.prompts[0]
        assert provider.systems[0] == prompts.SYSTEM_PROMPT

    def test_long_script_is_compressed(self) -> None:
        transcript = self._transcript(200)
        repaired_turns = [{"speaker": "Maya", "text": "word " * 100}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert result.word_count() == 100
        assert "Compress" in provider.prompts[0]

    def test_repair_prompt_and_result_carry_delivery_notes(self) -> None:
        transcript = Transcript(
            title="T",
            hosts=["Alex", "Maya"],
            turns=[Turn(speaker="Alex", text="w " * 50, delivery="wry")],
        )
        repaired_turns = [{"speaker": "Maya", "text": "word " * 100, "delivery": "calm"}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert "**Alex [wry]:**" in provider.prompts[0]
        assert "never count them" in provider.prompts[0]  # notes excluded from word math
        assert result.turns[0].delivery == "calm"

    def test_repair_prompt_says_stress_marks_are_spoken_and_kept(self) -> None:
        transcript = self._transcript(50)
        repaired_turns = [{"speaker": "Maya", "text": "word " * 100}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert "*word*" in provider.prompts[0]  # stress marks are spoken text, counted
        assert "stress marks" in provider.prompts[0]

    def test_worse_repair_is_discarded(self) -> None:
        transcript = self._transcript(80)
        repaired_turns = [{"speaker": "Maya", "text": "word " * 10}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert result is transcript


def _config_for(format_key: str) -> AppConfig:
    config = AppConfig()
    config.script.format = format_key
    return config


class TestEpisodeHosts:
    def test_two_speaker_formats_use_all_hosts(self) -> None:
        config = _config_for("debate")
        spec = pipeline.episode_format(config)
        assert [host.name for host in pipeline.episode_hosts(config, spec)] == ["Alex", "Maya"]

    def test_solo_defaults_to_the_first_host(self) -> None:
        config = _config_for("brief")
        spec = pipeline.episode_format(config)
        assert [host.name for host in pipeline.episode_hosts(config, spec)] == ["Alex"]

    def test_solo_host_setting_is_honored(self) -> None:
        config = _config_for("brief")
        config.script.solo_host = "Maya"
        spec = pipeline.episode_format(config)
        assert [host.name for host in pipeline.episode_hosts(config, spec)] == ["Maya"]

    def test_two_person_formats_trim_extra_hosts(self) -> None:
        third = HostSpec(name="Sam", gender="male", persona="the third voice")
        for key in ("debate", "critique"):
            config = _config_for(key)
            config.script.hosts = [*config.script.hosts, third]
            spec = pipeline.episode_format(config)
            names = [host.name for host in pipeline.episode_hosts(config, spec)]
            assert names == ["Alex", "Maya"]

    def test_deep_dive_keeps_every_configured_host(self) -> None:
        config = _config_for("deep-dive")
        config.script.hosts = [
            *config.script.hosts,
            HostSpec(name="Sam", gender="male", persona="the third voice"),
        ]
        spec = pipeline.episode_format(config)
        assert [host.name for host in pipeline.episode_hosts(config, spec)] == [
            "Alex",
            "Maya",
            "Sam",
        ]


class TestBriefFormat:
    def test_outline_prompt_carries_brief_shape(self) -> None:
        reply = json.dumps({"title": "T", "segments": [{"heading": "a", "target_words": 50}]})
        provider = _ScriptedProvider([reply])
        pipeline.build_outline(provider, _config_for("brief"), SOURCES, 50)
        assert provider.systems[0] == formats.FORMATS["brief"].system_prompt
        assert "Break the episode into 1-2 segments" in provider.prompts[0]
        assert formats.FORMATS["brief"].outline_brief in provider.prompts[0]

    def test_dialogue_is_solo_start_to_finish(self) -> None:
        config = _config_for("brief")
        outline = pipeline.build_outline(FakeProvider(), config, SOURCES, 120)
        transcript = pipeline.write_dialogue(FakeProvider(), config, SOURCES, outline)
        assert transcript.hosts == ["Alex"]
        assert transcript.format == "brief"
        assert {turn.speaker for turn in transcript.turns} == {"Alex"}

    def test_solo_schema_and_request_phrasing(self) -> None:
        config = _config_for("brief")
        outline = Outline(title="T", segments=[OutlineSegment(heading="only", target_words=60)])
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": "hello"}]})
        provider = _ScriptedProvider([reply])
        pipeline.write_dialogue(provider, config, SOURCES, outline)
        assert '"enum": ["Alex"]' in json.dumps(provider.schemas[0])
        assert "Alex speaking alone, directly to the listener" in provider.prompts[0]

    def test_undershoot_is_accepted_without_an_llm_call(self) -> None:
        transcript = Transcript(
            title="T", hosts=["Alex"], turns=[Turn(speaker="Alex", text="w " * 100)]
        )
        provider = _ScriptedProvider([])
        result = pipeline.ensure_length(provider, _config_for("brief"), transcript, 300)
        assert result is transcript
        assert provider.schemas == []

    def test_overshoot_is_still_compressed(self) -> None:
        transcript = Transcript(
            title="T", hosts=["Alex"], turns=[Turn(speaker="Alex", text="w " * 500)]
        )
        repaired = [{"speaker": "Alex", "text": "word " * 300}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired})])
        result = pipeline.ensure_length(provider, _config_for("brief"), transcript, 300)
        assert result.word_count() == 300
        assert "Compress" in provider.prompts[0]


class TestDebateFormat:
    def test_outline_schema_requires_a_stance_per_host(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [{"heading": "a", "target_words": 50}],
                "host_angles": {"Alex": "for", "Maya": "against"},
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, _config_for("debate"), SOURCES, 50)
        schema = provider.schemas[0]
        assert schema is not None
        rendered = json.dumps(schema)
        assert '"host_angles"' in rendered
        assert '"required": ["Alex", "Maya"]' in rendered
        top_required = schema.get("required")
        assert isinstance(top_required, list)
        assert "host_angles" in top_required  # native-schema providers must emit it
        assert outline.host_angles == {"Alex": "for", "Maya": "against"}

    def test_missing_stances_fail_loudly(self) -> None:
        reply = json.dumps({"title": "T", "segments": [{"heading": "a", "target_words": 50}]})
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match="did not assign a debate stance"):
            pipeline.build_outline(provider, _config_for("debate"), SOURCES, 50)

    def test_fake_provider_assigns_stances(self) -> None:
        outline = pipeline.build_outline(FakeProvider(), _config_for("debate"), SOURCES, 300)
        assert set(outline.host_angles) == {"Alex", "Maya"}
        assert all(stance for stance in outline.host_angles.values())

    def test_stances_reach_every_dialogue_request(self) -> None:
        config = _config_for("debate")
        outline = Outline(
            title="T",
            segments=[
                OutlineSegment(heading="one", target_words=60),
                OutlineSegment(heading="two", target_words=60),
            ],
            host_angles={"Alex": "argues for", "Maya": "argues against"},
        )
        reply = json.dumps({"turns": [{"speaker": "Alex", "text": "hello"}]})
        provider = _ScriptedProvider([reply, reply])
        pipeline.write_dialogue(provider, config, SOURCES, outline)
        for prompt in provider.prompts:
            assert "Assigned stance (argue this side all episode): argues for" in prompt
            assert "Assigned stance (argue this side all episode): argues against" in prompt

    def test_unknown_stance_keys_are_dropped(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [{"heading": "a", "target_words": 50}],
                "host_angles": {"Narrator": "meta", "Alex": "for", "Maya": "against"},
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, _config_for("debate"), SOURCES, 50)
        assert outline.host_angles == {"Alex": "for", "Maya": "against"}

    def test_blank_stance_counts_as_missing(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [{"heading": "a", "target_words": 50}],
                "host_angles": {"Alex": "   ", "Maya": "against"},
            }
        )
        provider = _ScriptedProvider([reply])
        with pytest.raises(ScriptError, match="did not assign a debate stance to Alex"):
            pipeline.build_outline(provider, _config_for("debate"), SOURCES, 50)

    def test_case_mangled_stance_keys_are_normalized(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [{"heading": "a", "target_words": 50}],
                "host_angles": {"alex ": "for", "MAYA [the guide]": "against"},
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, _config_for("debate"), SOURCES, 50)
        assert outline.host_angles == {"Alex": "for", "Maya": "against"}

    def test_volunteered_angles_are_dropped_outside_debate(self) -> None:
        reply = json.dumps(
            {
                "title": "T",
                "segments": [{"heading": "a", "target_words": 50}],
                "host_angles": {"Alex": "pro", "Maya": "con"},
            }
        )
        provider = _ScriptedProvider([reply])
        outline = pipeline.build_outline(provider, AppConfig(), SOURCES, 50)
        assert outline.host_angles == {}
        turn = json.dumps({"turns": [{"speaker": "Alex", "text": "hi"}]})
        dialogue_provider = _ScriptedProvider([turn])
        pipeline.write_dialogue(dialogue_provider, AppConfig(), SOURCES, outline)
        assert "Assigned stance" not in dialogue_provider.prompts[0]

    def test_non_debate_outline_schema_has_no_angles(self) -> None:
        reply = json.dumps({"title": "T", "segments": [{"heading": "a", "target_words": 50}]})
        provider = _ScriptedProvider([reply])
        pipeline.build_outline(provider, AppConfig(), SOURCES, 50)
        assert "host_angles" not in json.dumps(provider.schemas[0])

    def test_expansion_carries_the_no_padding_rule(self) -> None:
        transcript = Transcript(
            title="T", hosts=["Alex", "Maya"], turns=[Turn(speaker="Alex", text="w " * 50)]
        )
        repaired = [{"speaker": "Maya", "text": "word " * 100}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired})])
        pipeline.ensure_length(provider, _config_for("debate"), transcript, 100)
        assert formats.FORMATS["debate"].extend_guidance in provider.prompts[0]

    def test_polish_uses_the_adversarial_brief(self) -> None:
        polished = [{"speaker": "Maya", "text": "sharper line"}]
        provider = _ScriptedProvider([json.dumps({"turns": polished})])
        transcript = Transcript(
            title="T",
            hosts=["Alex", "Maya"],
            turns=[Turn(speaker="Alex", text="one two three")],
            format="debate",
        )
        pipeline.polish_dialogue(provider, _config_for("debate"), transcript)
        assert provider.systems[0] == formats.FORMATS["debate"].system_prompt
        assert formats.FORMATS["debate"].polish_brief in provider.prompts[0]


class TestCritiqueFormat:
    def test_review_source_drafts_then_reflects(self) -> None:
        review = pipeline.review_source(FakeProvider(), _config_for("critique"), SOURCES)
        assert len(review.findings) >= 3
        assert all(finding.anchor for finding in review.findings)

    def test_reflection_pass_sees_draft_and_document(self) -> None:
        draft = {
            "strengths": ["s"],
            "findings": [
                {"summary": "a", "anchor": "b", "detail": "c", "suggestion": "d"},
                {"summary": "e", "anchor": "f", "detail": "g", "suggestion": "h"},
                {"summary": "i", "anchor": "j", "detail": "k", "suggestion": "l"},
            ],
            "questions": [],
        }
        provider = _ScriptedProvider([json.dumps(draft), json.dumps(draft)])
        pipeline.review_source(provider, _config_for("critique"), SOURCES)
        assert provider.systems == [formats.CRITIQUE_REVIEW_PROMPT] * 2
        assert "Re-examine each draft finding" in provider.prompts[1]
        assert SOURCES in provider.prompts[1]

    def test_outline_dialogue_ifies_the_review(self) -> None:
        config = _config_for("critique")
        review = pipeline.review_source(FakeProvider(), config, SOURCES)
        reply = json.dumps({"title": "T", "segments": [{"heading": "a", "target_words": 50}]})
        provider = _ScriptedProvider([reply])
        pipeline.build_outline(provider, config, SOURCES, 50, review=review)
        assert "A structured review of the material" in provider.prompts[0]
        assert review.findings[0].summary in provider.prompts[0]
        assert formats.FORMATS["critique"].outline_brief in provider.prompts[0]

    def test_transcript_records_the_format(self) -> None:
        config = _config_for("critique")
        outline = pipeline.build_outline(FakeProvider(), config, SOURCES, 300)
        transcript = pipeline.write_dialogue(FakeProvider(), config, SOURCES, outline)
        assert transcript.format == "critique"
        assert transcript.hosts == ["Alex", "Maya"]
