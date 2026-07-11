"""Outline → per-segment dialogue → length repair (the generate stage)."""

import re
from collections.abc import Callable
from typing import cast

from podcast.config import AppConfig
from podcast.errors import ScriptError
from podcast.llm.base import ChatMessage, ChatProvider, system, user
from podcast.llm.structured import complete_structured
from podcast.script.budget import within_tolerance
from podcast.script.markdown import turn_to_line
from podcast.script.models import DialogueChunk, Outline, Transcript, Turn
from podcast.script.prompts import (
    CONTINUING_POSITION,
    FINAL_POSITION,
    OPENING_POSITION,
    OUTLINE_BRIEF,
    SYSTEM_PROMPT,
)

_CONTEXT_TURNS = 2
_NOTED_SPEAKER_RE = re.compile(r"^(.*?)\s*\[.*\]$")


def _hosts_brief(config: AppConfig) -> str:
    return "\n".join(
        f"- {host.name} ({host.gender}): {host.persona}" for host in config.script.hosts
    )


def _rescale_outline(outline: Outline, budget_words: int) -> Outline:
    """Force segment budgets to sum exactly to the episode budget."""
    total = outline.total_words()
    if total == budget_words:
        return outline
    scaled = [
        max(1, round(segment.target_words * budget_words / total)) for segment in outline.segments
    ]
    scaled[-1] += budget_words - sum(scaled)
    segments = [
        segment.model_copy(update={"target_words": words})
        for segment, words in zip(outline.segments, scaled, strict=True)
    ]
    return outline.model_copy(update={"segments": segments})


def build_outline(
    provider: ChatProvider, config: AppConfig, sources_markdown: str, budget_words: int
) -> Outline:
    """Plan the episode: titled segments whose word targets sum to the budget."""
    messages = [
        system(SYSTEM_PROMPT),
        user(
            "Plan a podcast episode from these sources.\n\n"
            f"{sources_markdown}\n\n"
            f"Hosts:\n{_hosts_brief(config)}\n\n"
            f"The full dialogue must total approximately {budget_words} words. "
            "Break the episode into 3-6 segments; give each a heading, notes on what "
            "to cover (with source references), and a target_words share. The "
            f"target_words values must sum to {budget_words}. {OUTLINE_BRIEF}"
        ),
    ]
    outline = complete_structured(
        provider,
        messages,
        Outline,
        temperature=config.llm.outline_temperature,
        max_retries=config.llm.max_retries,
    )
    return _rescale_outline(outline, budget_words)


def _dialogue_schema(host_names: list[str]) -> dict[str, object]:
    """DialogueChunk's schema with the speaker field constrained to real hosts.

    The nested keys are guaranteed by our own model definition; a KeyError here
    means the DialogueChunk/Turn models changed shape.
    """
    schema: dict[str, object] = DialogueChunk.model_json_schema()
    definitions = cast("dict[str, object]", schema["$defs"])
    turn = cast("dict[str, object]", definitions["Turn"])
    properties = cast("dict[str, object]", turn["properties"])
    speaker = cast("dict[str, object]", properties["speaker"])
    speaker["enum"] = list(host_names)
    return schema


def _resolve_speaker(name: str, host_names: list[str]) -> str:
    """Match a host name case-insensitively; a model echoing the quoted-context
    notation (`Alex [warm]`) resolves to the bare host instead of aborting the run."""
    candidate = name.strip()
    suffix = _NOTED_SPEAKER_RE.match(candidate)
    if suffix is not None:
        candidate = suffix.group(1)
    for host in host_names:
        if host.lower() == candidate.lower():
            return host
    raise ScriptError(f"the model used unknown speaker {name!r} (hosts: {', '.join(host_names)})")


def _dialogue_request(
    provider: ChatProvider,
    config: AppConfig,
    messages: list[ChatMessage],
    host_names: list[str],
) -> list[Turn]:
    chunk = complete_structured(
        provider,
        messages,
        DialogueChunk,
        temperature=config.llm.dialogue_temperature,
        max_retries=config.llm.max_retries,
        schema=_dialogue_schema(host_names),
    )
    return [
        Turn(
            speaker=_resolve_speaker(turn.speaker, host_names),
            text=turn.text,
            delivery=turn.delivery,
        )
        for turn in chunk.turns
        if turn.text.strip()
    ]


def write_dialogue(
    provider: ChatProvider,
    config: AppConfig,
    sources_markdown: str,
    outline: Outline,
    on_segment: Callable[[int], None] | None = None,
) -> Transcript:
    """Generate every segment's turns, threading recent turns through for continuity."""
    host_names = [host.name for host in config.script.hosts]
    turns: list[Turn] = []
    for index, segment in enumerate(outline.segments):
        previous = "\n".join(turn_to_line(turn) for turn in turns[-_CONTEXT_TURNS:])
        position = OPENING_POSITION if index == 0 else CONTINUING_POSITION
        if index == len(outline.segments) - 1:
            position += FINAL_POSITION
        messages = [
            system(SYSTEM_PROMPT),
            user(
                f"Sources:\n\n{sources_markdown}\n\n"
                f"Episode: {outline.title}\n"
                f"Hosts:\n{_hosts_brief(config)}\n\n"
                f"Segment {index + 1} of {len(outline.segments)}: {segment.heading}\n"
                f"Notes: {segment.notes}\n"
                f"You are {position}.\n"
                + (f"The conversation so far ended with:\n{previous}\n" if previous else "")
                + f"Write this segment's dialogue: approximately {segment.target_words} "
                "words, alternating naturally between the hosts."
            ),
        ]
        segment_turns = _dialogue_request(provider, config, messages, host_names)
        if not segment_turns:
            raise ScriptError(f"segment {index + 1} ({segment.heading!r}) produced no dialogue")
        turns.extend(segment_turns)
        if on_segment is not None:
            on_segment(index)
    return Transcript(title=outline.title, hosts=host_names, turns=turns)


def ensure_length(
    provider: ChatProvider,
    config: AppConfig,
    transcript: Transcript,
    budget_words: int,
) -> Transcript:
    """One repair pass when the script is >tolerance off target; keeps the closer version."""
    actual = transcript.word_count()
    if within_tolerance(actual, budget_words, config.script.length_tolerance):
        return transcript
    direction = "Expand" if actual < budget_words else "Compress"
    script_text = "\n".join(turn_to_line(turn) for turn in transcript.turns)
    messages = [
        system(SYSTEM_PROMPT),
        user(
            f"This podcast script is {actual} words but must be approximately "
            f"{budget_words} words. {direction} it to hit the target while keeping "
            "the same hosts, structure, facts, and natural flow. Both word counts "
            "cover spoken text only: the bracketed delivery notes on some lines are "
            "performance metadata — keep or adjust them, but never count them.\n\n"
            f"{script_text}"
        ),
    ]
    repaired_turns = _dialogue_request(provider, config, messages, transcript.hosts)
    repaired = transcript.model_copy(update={"turns": repaired_turns})
    if abs(repaired.word_count() - budget_words) < abs(actual - budget_words):
        return repaired
    return transcript
