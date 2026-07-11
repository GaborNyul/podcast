"""Tests for podcast.script.pipeline (fake + scripted providers)."""

import json
from collections.abc import Mapping, Sequence

import pytest

from podcast.config import AppConfig
from podcast.errors import ScriptError
from podcast.llm.base import ChatMessage
from podcast.llm.fake import FakeProvider
from podcast.script import pipeline, prompts
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

    def test_worse_repair_is_discarded(self) -> None:
        transcript = self._transcript(80)
        repaired_turns = [{"speaker": "Maya", "text": "word " * 10}]
        provider = _ScriptedProvider([json.dumps({"turns": repaired_turns})])
        result = pipeline.ensure_length(provider, AppConfig(), transcript, 100)
        assert result is transcript
