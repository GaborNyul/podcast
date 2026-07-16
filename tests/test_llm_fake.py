# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.llm.fake."""

import json

from podcast.llm.base import user
from podcast.llm.fake import FakeProvider

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "segments": {"type": "array"}},
}

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"enum": ["Alex", "Maya"]},
                    "text": {"type": "string"},
                },
            },
        }
    },
}

DIALOGUE_SCHEMA_WITH_REF = {
    "$defs": {"Speaker": {"enum": ["Ági", "Ödön"]}},
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"$ref": "#/$defs/Speaker"},
                    "text": {"type": "string"},
                },
            },
        }
    },
}


def _word_count(text: str) -> int:
    return len(text.split())


class TestComplete:
    def test_no_schema_returns_plain_text(self) -> None:
        assert FakeProvider().complete([user("hi")], temperature=0.5) == "Understood."

    def test_unknown_schema_returns_empty_object(self) -> None:
        reply = FakeProvider().complete(
            [user("hi")], temperature=0.5, json_schema={"type": "object"}
        )
        assert reply == "{}"

    def test_outline_totals_requested_words(self) -> None:
        reply = FakeProvider().complete(
            [user("Plan a script of approximately 400 words.")],
            temperature=0.3,
            json_schema=OUTLINE_SCHEMA,
        )
        outline = json.loads(reply)
        assert outline["title"]
        totals = [segment["target_words"] for segment in outline["segments"]]
        assert sum(totals) == 400

    def test_outline_defaults_without_word_hint(self) -> None:
        reply = FakeProvider().complete(
            [user("Plan a script.")], temperature=0.3, json_schema=OUTLINE_SCHEMA
        )
        outline = json.loads(reply)
        assert sum(segment["target_words"] for segment in outline["segments"]) == 300

    def test_dialogue_hits_word_target_exactly(self) -> None:
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 120 words.")],
            temperature=0.8,
            json_schema=DIALOGUE_SCHEMA,
        )
        dialogue = json.loads(reply)
        total = sum(_word_count(turn["text"]) for turn in dialogue["turns"])
        assert total == 120

    def test_dialogue_alternates_schema_speakers(self) -> None:
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 90 words.")],
            temperature=0.8,
            json_schema=DIALOGUE_SCHEMA,
        )
        speakers = [turn["speaker"] for turn in json.loads(reply)["turns"]]
        assert set(speakers) == {"Alex", "Maya"}
        assert all(first != second for first, second in zip(speakers, speakers[1:], strict=False))

    def test_dialogue_resolves_ref_speakers(self) -> None:
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=DIALOGUE_SCHEMA_WITH_REF,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Ági", "Ödön"}

    def test_speaker_without_enum_or_ref_falls_back(self) -> None:
        schema = {
            "type": "object",
            "properties": {"turns": {"items": {"properties": {"speaker": {"type": "string"}}}}},
        }
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=schema,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Host A", "Host B"}

    def test_ref_to_missing_def_falls_back(self) -> None:
        schema = {
            "$defs": {},
            "type": "object",
            "properties": {
                "turns": {"items": {"properties": {"speaker": {"$ref": "#/$defs/Ghost"}}}}
            },
        }
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=schema,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Host A", "Host B"}

    def test_ref_target_without_enum_falls_back(self) -> None:
        schema = {
            "$defs": {"Speaker": {"type": "string"}},
            "type": "object",
            "properties": {
                "turns": {"items": {"properties": {"speaker": {"$ref": "#/$defs/Speaker"}}}}
            },
        }
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=schema,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Host A", "Host B"}

    def test_speaker_enum_found_through_anyof_list(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "turns": {
                    "anyOf": [{"items": {"properties": {"speaker": {"enum": ["Left", "Right"]}}}}]
                }
            },
        }
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=schema,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Left", "Right"}

    def test_dialogue_without_enum_uses_fallback_speakers(self) -> None:
        schema = {"type": "object", "properties": {"turns": {"type": "array"}}}
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 60 words.")],
            temperature=0.8,
            json_schema=schema,
        )
        speakers = {turn["speaker"] for turn in json.loads(reply)["turns"]}
        assert speakers == {"Host A", "Host B"}

    def test_dialogue_mixes_annotated_and_neutral_deliveries(self) -> None:
        reply = FakeProvider().complete(
            [user("Write dialogue of approximately 120 words.")],
            temperature=0.8,
            json_schema=DIALOGUE_SCHEMA,
        )
        deliveries = [turn["delivery"] for turn in json.loads(reply)["turns"]]
        assert "warm, curious" in deliveries
        assert "" in deliveries

    def test_deterministic_across_calls(self) -> None:
        messages = [user("Write dialogue of approximately 100 words.")]
        first = FakeProvider().complete(messages, temperature=0.8, json_schema=DIALOGUE_SCHEMA)
        second = FakeProvider().complete(messages, temperature=0.1, json_schema=DIALOGUE_SCHEMA)
        assert first == second
