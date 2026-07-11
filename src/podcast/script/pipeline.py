"""Outline → per-segment dialogue → length repair (the generate stage).

Every stage reads its prompts and episode shape from the FormatSpec resolved
off `config.script.format` (ADR 0013); the default deep-dive format reproduces
the pre-format pipeline byte for byte.
"""

import re
from collections.abc import Callable
from typing import cast

from podcast.config import AppConfig, HostSpec
from podcast.errors import ScriptError
from podcast.llm.base import ChatMessage, ChatProvider, system, user
from podcast.llm.structured import complete_structured
from podcast.script import formats
from podcast.script.budget import within_tolerance
from podcast.script.formats import FormatSpec
from podcast.script.markdown import turn_to_line
from podcast.script.models import (
    CritiqueReview,
    DialogueChunk,
    Outline,
    Transcript,
    Turn,
)

_CONTEXT_TURNS = 2
_NOTED_SPEAKER_RE = re.compile(r"^(.*?)\s*\[.*\]$")


def episode_format(config: AppConfig) -> FormatSpec:
    """The FormatSpec the pipeline runs under."""
    return formats.resolve(config.script.format)


def episode_hosts(config: AppConfig, spec: FormatSpec) -> list[HostSpec]:
    """The hosts this format actually uses: all of them (speakers=None), the
    solo narrator, or the first `speakers` configured hosts — debate/critique
    define exactly two roles, so extra configured hosts sit those formats out."""
    hosts = list(config.script.hosts)
    if spec.speakers is None:
        return hosts
    if spec.speakers == 1:
        if config.script.solo_host is not None:
            return [host for host in hosts if host.name == config.script.solo_host]
        return hosts[:1]
    return hosts[: spec.speakers]


def _hosts_brief(hosts: list[HostSpec], angles: dict[str, str]) -> str:
    lines: list[str] = []
    for host in hosts:
        line = f"- {host.name} ({host.gender}): {host.persona}"
        stance = angles.get(host.name, "")
        if stance:
            line += f"\n  Assigned stance (argue this side all episode): {stance}"
        lines.append(line)
    return "\n".join(lines)


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


def _outline_schema(spec: FormatSpec, host_names: list[str]) -> dict[str, object]:
    """Outline's schema, with host_angles required for debate and absent otherwise.

    The nested keys are guaranteed by our own model definition; a KeyError here
    means the Outline model changed shape.
    """
    schema: dict[str, object] = Outline.model_json_schema()
    properties = cast("dict[str, object]", schema["properties"])
    if spec.assigns_stances:
        properties["host_angles"] = {
            "type": "object",
            "description": (
                "The stance each host argues for the whole episode, keyed by host "
                "name — one entry per host, derived from where the sources "
                "genuinely pull apart."
            ),
            "properties": {name: {"type": "string"} for name in host_names},
            "required": list(host_names),
            "additionalProperties": False,
        }
        required = cast("list[str]", schema.setdefault("required", []))
        if "host_angles" not in required:
            required.append("host_angles")
    else:
        properties.pop("host_angles", None)
    return schema


def review_source(
    provider: ChatProvider, config: AppConfig, sources_markdown: str
) -> CritiqueReview:
    """The critique format's pre-outline review: draft, then one reflection pass
    that re-checks every finding against the material (hallucinated-flaw guard)."""
    spec = episode_format(config)
    messages = [
        system(spec.review_prompt),
        user(f"Review this document.\n\n{sources_markdown}"),
    ]
    draft = complete_structured(
        provider,
        messages,
        CritiqueReview,
        temperature=config.llm.outline_temperature,
        max_retries=config.llm.max_retries,
    )
    reflection = [
        system(spec.review_prompt),
        user(
            f"The document under review:\n\n{sources_markdown}\n\n"
            f"A draft review:\n{draft.model_dump_json(indent=2)}\n\n"
            "Re-examine each draft finding against the document. Drop any finding "
            "whose anchor does not hold, sharpen the ones that survive, and return "
            "the final review."
        ),
    ]
    return complete_structured(
        provider,
        reflection,
        CritiqueReview,
        temperature=config.llm.outline_temperature,
        max_retries=config.llm.max_retries,
    )


def build_outline(
    provider: ChatProvider,
    config: AppConfig,
    sources_markdown: str,
    budget_words: int,
    review: CritiqueReview | None = None,
) -> Outline:
    """Plan the episode: titled segments whose word targets sum to the budget."""
    spec = episode_format(config)
    hosts = episode_hosts(config, spec)
    host_names = [host.name for host in hosts]
    low, high = spec.segment_range
    review_block = (
        "A structured review of the material (the episode dialogue-ifies these "
        f"findings):\n{review.model_dump_json(indent=2)}\n\n"
        if review is not None
        else ""
    )
    messages = [
        system(spec.system_prompt),
        user(
            "Plan a podcast episode from these sources.\n\n"
            f"{sources_markdown}\n\n"
            f"{review_block}"
            f"Hosts:\n{_hosts_brief(hosts, {})}\n\n"
            f"The full dialogue must total approximately {budget_words} words. "
            f"Break the episode into {low}-{high} segments; give each a heading, notes on what "
            "to cover (with source references), and a target_words share. The "
            f"target_words values must sum to {budget_words}. {spec.outline_brief}"
        ),
    ]
    outline = complete_structured(
        provider,
        messages,
        Outline,
        temperature=config.llm.outline_temperature,
        max_retries=config.llm.max_retries,
        schema=_outline_schema(spec, host_names),
    )
    if spec.assigns_stances:
        outline = outline.model_copy(
            update={"host_angles": _normalized_angles(outline.host_angles, host_names)}
        )
    elif outline.host_angles:
        # A model may volunteer stances the schema never asked for; formats
        # without stance assignment must not let them leak into later prompts.
        outline = outline.model_copy(update={"host_angles": {}})
    return _rescale_outline(outline, budget_words)


def _normalized_angles(angles: dict[str, str], host_names: list[str]) -> dict[str, str]:
    """Resolve LLM-emitted stance keys onto real host names (case-insensitive,
    tolerant of a bracketed-note suffix); every host must end up with a stance."""
    resolved: dict[str, str] = {}
    for key, stance in angles.items():
        try:
            name = _resolve_speaker(key, host_names)
        except ScriptError:
            continue
        if stance.strip():
            resolved.setdefault(name, stance.strip())
    missing = [name for name in host_names if name not in resolved]
    if missing:
        raise ScriptError(
            f"the outline did not assign a debate stance to {', '.join(missing)}; "
            "re-run generate (the model must fill host_angles for every host)"
        )
    return resolved


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


def _segment_request(host_names: list[str], target_words: int) -> str:
    if len(host_names) == 1:
        return (
            f"Write this segment's narration: approximately {target_words} words, "
            f"{host_names[0]} speaking alone, directly to the listener."
        )
    return (
        f"Write this segment's dialogue: approximately {target_words} "
        "words, alternating naturally between the hosts."
    )


def write_dialogue(
    provider: ChatProvider,
    config: AppConfig,
    sources_markdown: str,
    outline: Outline,
    on_segment: Callable[[int], None] | None = None,
) -> Transcript:
    """Generate every segment's turns, threading recent turns through for continuity."""
    spec = episode_format(config)
    hosts = episode_hosts(config, spec)
    host_names = [host.name for host in hosts]
    hosts_brief = _hosts_brief(hosts, outline.host_angles)
    turns: list[Turn] = []
    for index, segment in enumerate(outline.segments):
        previous = "\n".join(turn_to_line(turn) for turn in turns[-_CONTEXT_TURNS:])
        position = spec.opening_position if index == 0 else spec.continuing_position
        if index == len(outline.segments) - 1:
            position += spec.final_position
        messages = [
            system(spec.system_prompt),
            user(
                f"Sources:\n\n{sources_markdown}\n\n"
                f"Episode: {outline.title}\n"
                f"Hosts:\n{hosts_brief}\n\n"
                f"Segment {index + 1} of {len(outline.segments)}: {segment.heading}\n"
                f"Notes: {segment.notes}\n"
                f"You are {position}.\n"
                + (f"The conversation so far ended with:\n{previous}\n" if previous else "")
                + _segment_request(host_names, segment.target_words)
            ),
        ]
        segment_turns = _dialogue_request(provider, config, messages, host_names)
        if not segment_turns:
            raise ScriptError(f"segment {index + 1} ({segment.heading!r}) produced no dialogue")
        turns.extend(segment_turns)
        if on_segment is not None:
            on_segment(index)
    return Transcript(title=outline.title, hosts=host_names, turns=turns, format=spec.key)


def polish_dialogue(
    provider: ChatProvider, config: AppConfig, transcript: Transcript
) -> Transcript:
    """One whole-script rewrite for radio texture (disfluencies, interruptions,
    sharper delivery notes) — NotebookLM's dedicated disfluency pass (ADR 0011)."""
    if not config.script.polish_pass:
        return transcript
    spec = episode_format(config)
    script_text = "\n".join(turn_to_line(turn) for turn in transcript.turns)
    messages = [
        system(spec.system_prompt),
        user(
            f"{spec.polish_brief}, and keep the total at approximately "
            f"{transcript.word_count()} words:\n\n{script_text}"
        ),
    ]
    polished_turns = _dialogue_request(provider, config, messages, transcript.hosts)
    if not polished_turns:
        return transcript
    return transcript.model_copy(update={"turns": polished_turns})


def ensure_length(
    provider: ChatProvider,
    config: AppConfig,
    transcript: Transcript,
    budget_words: int,
) -> Transcript:
    """One repair pass when the script is >tolerance off target; keeps the closer version.

    Ceiling formats (brief) accept any undershoot — short is the point — and
    only compress overshoot."""
    spec = episode_format(config)
    actual = transcript.word_count()
    if spec.length_mode == "ceiling" and actual <= budget_words:
        return transcript
    if within_tolerance(actual, budget_words, config.script.length_tolerance):
        return transcript
    direction = "Expand" if actual < budget_words else "Compress"
    guidance = f" {spec.extend_guidance}" if direction == "Expand" and spec.extend_guidance else ""
    script_text = "\n".join(turn_to_line(turn) for turn in transcript.turns)
    messages = [
        system(spec.system_prompt),
        user(
            f"This podcast script is {actual} words but must be approximately "
            f"{budget_words} words. {direction} it to hit the target while keeping "
            f"the same hosts, structure, facts, and natural flow.{guidance} Both word counts "
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
