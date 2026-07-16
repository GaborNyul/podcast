# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Deterministic offline provider for tests and offline end-to-end runs.

Contract with the script pipeline (kept in sync with `podcast.script`): prompts
state word targets as "approximately N words"; the outline schema exposes a
"segments" property (plus, for the debate format, a "host_angles" object whose
per-host property names carry the host names to assign stances to); the critique
review schema exposes a "findings" property; the dialogue schema exposes a
"turns" property whose `speaker` field is an enum of host names and whose
`delivery` field carries the per-turn performance note (this provider emits a
deterministic mix of annotated and neutral turns so the delivery path is
exercised offline).
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from podcast.llm.base import ChatMessage

_WORDS_RE = re.compile(r"approximately (\d+) words")
_DEFAULT_WORDS = 300
_WORDS_PER_TURN = 30

_DELIVERIES = ("warm, curious", "", "excited, picking up speed", "")

_VOCAB = [
    "the",
    "sources",
    "explain",
    "this",
    "idea",
    "in",
    "plain",
    "terms",
    "and",
    "connect",
    "it",
    "to",
    "a",
    "bigger",
    "picture",
    "so",
    "a",
    "listener",
    "can",
    "follow",
    "each",
    "step",
    "with",
    "a",
    "concrete",
    "example",
    "in",
    "mind",
]

_HEADINGS = ("Setting the Scene", "The Core Ideas", "What It All Means")


def _prose(word_count: int, offset: int = 0) -> str:
    """Deterministic filler text with exactly `word_count` words."""
    words = [_VOCAB[(offset + index) % len(_VOCAB)] for index in range(word_count)]
    sentences: list[str] = []
    for start in range(0, len(words), 10):
        chunk = words[start : start + 10]
        sentences.append(chunk[0].capitalize() + " " + " ".join(chunk[1:]) + ".")
    return " ".join(sentences)


def _requested_words(messages: Sequence[ChatMessage]) -> int:
    for message in reversed(messages):
        match = _WORDS_RE.search(message.content)
        if match:
            return int(match.group(1))
    return _DEFAULT_WORDS


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _enum_values(node: Mapping[str, object]) -> list[str] | None:
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return [str(value) for value in cast("list[object]", enum)]
    return None


def _speaker_enum(schema: Mapping[str, object]) -> list[str]:
    """Depth-first search for the `speaker` property's enum values."""
    stack: list[object] = [schema]
    while stack:
        node = _as_mapping(stack.pop())
        if node is None:
            continue
        speaker = _as_mapping(node.get("speaker"))
        if speaker is not None:
            enum = _enum_values(speaker)
            if enum is not None:
                return enum
            ref = speaker.get("$ref")
            defs = _as_mapping(schema.get("$defs"))
            if isinstance(ref, str) and defs is not None:
                target = _as_mapping(defs.get(ref.rsplit("/", 1)[-1]))
                if target is not None:
                    enum = _enum_values(target)
                    if enum is not None:
                        return enum
        for value in node.values():
            stack.append(value)
            if isinstance(value, list):
                stack.extend(cast("list[object]", value))
    return ["Host A", "Host B"]


class FakeProvider:
    """Schema-aware canned responses; fully deterministic, zero network."""

    name = "fake"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        del temperature  # deterministic by design
        if json_schema is None:
            return "Understood."
        properties = _as_mapping(json_schema.get("properties"))
        keys: set[str] = set(properties) if properties is not None else set()
        if "segments" in keys:
            return self._outline(messages, properties)
        if "findings" in keys:
            return self._review()
        if "turns" in keys:
            return self._dialogue(messages, json_schema)
        return "{}"

    def _outline(
        self, messages: Sequence[ChatMessage], properties: Mapping[str, object] | None
    ) -> str:
        total = _requested_words(messages)
        share = total // len(_HEADINGS)
        remainder = total - share * len(_HEADINGS)
        segments = [
            {
                "heading": heading,
                "target_words": share + (remainder if index == len(_HEADINGS) - 1 else 0),
                "notes": f"Cover {heading.lower()} using only facts from the sources.",
            }
            for index, heading in enumerate(_HEADINGS)
        ]
        payload: dict[str, object] = {"title": "A Guided Tour of the Sources", "segments": segments}
        angles = _as_mapping(properties.get("host_angles")) if properties is not None else None
        if angles is not None:
            host_names = _as_mapping(angles.get("properties")) or {}
            stances = (
                "argues the sources support this reading",
                "argues the sources leave this open",
            )
            payload["host_angles"] = {
                name: stances[index % len(stances)] for index, name in enumerate(host_names)
            }
        return json.dumps(payload)

    def _review(self) -> str:
        findings = [
            {
                "summary": f"Claim {index} lacks supporting evidence.",
                "anchor": f"The sources state claim {index} without a comparison.",
                "detail": "An unsupported claim weakens the argument.",
                "suggestion": f"Add the missing comparison for claim {index}.",
            }
            for index in range(1, 4)
        ]
        return json.dumps(
            {
                "strengths": ["The structure follows the sources closely."],
                "findings": findings,
                "questions": ["What would falsify the central claim?"],
            }
        )

    def _dialogue(self, messages: Sequence[ChatMessage], json_schema: Mapping[str, object]) -> str:
        target = _requested_words(messages)
        speakers = _speaker_enum(json_schema)
        turn_count = max(2, target // _WORDS_PER_TURN)
        base = target // turn_count
        remainder = target - base * turn_count
        turns = [
            {
                "speaker": speakers[index % len(speakers)],
                "text": _prose(
                    base + (remainder if index == turn_count - 1 else 0), offset=index * 7
                ),
                "delivery": _DELIVERIES[index % len(_DELIVERIES)],
            }
            for index in range(turn_count)
        ]
        return json.dumps({"turns": turns})
