"""Transcript ⇄ script.md, the lossless contract between generate and synthesize.

Format: a front-matter block with JSON-encoded values, then exactly one line per
turn (`**Host:** text`). Turn text is whitespace-normalized on write so the
round-trip is byte-stable and hand edits stay unambiguous.
"""

import json
import re

from podcast.errors import ScriptError
from podcast.script.models import Transcript, Turn

_EDIT_HINT = (
    "<!-- Edit freely: one line per turn, formatted `**Host:** text`. "
    "Keep host names from the list above; then run `podcast synthesize`. -->"
)
_TURN_RE = re.compile(r"^\*\*(.+?):\*\* ?(.*)$")


def normalize_turn_text(text: str) -> str:
    """Collapse all whitespace so every turn serializes to a single line."""
    return " ".join(text.split())


def transcript_to_markdown(transcript: Transcript) -> str:
    lines = [
        "---",
        f"title: {json.dumps(transcript.title)}",
        f"hosts: {json.dumps(transcript.hosts)}",
        "---",
        "",
        _EDIT_HINT,
        "",
    ]
    for turn in transcript.turns:
        lines.append(f"**{turn.speaker}:** {normalize_turn_text(turn.text)}")
        lines.append("")
    return "\n".join(lines)


def _parse_front_matter(lines: list[str]) -> tuple[str, list[str], int]:
    if not lines or lines[0].strip() != "---":
        raise ScriptError("script.md must start with a --- front-matter block")
    title: str | None = None
    hosts: list[str] | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            if title is None or hosts is None:
                raise ScriptError("front matter must define both title and hosts")
            return title, hosts, index + 1
        key, _, raw_value = line.partition(":")
        try:
            value: object = json.loads(raw_value.strip())
        except json.JSONDecodeError as exc:
            raise ScriptError(f"front matter value for {key.strip()!r} is not JSON") from exc
        if key.strip() == "title" and isinstance(value, str):
            title = value
        elif key.strip() == "hosts" and isinstance(value, list):
            hosts = [str(item) for item in value]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    raise ScriptError("unterminated front-matter block in script.md")


def markdown_to_transcript(text: str) -> Transcript:
    """Parse script.md; raises ScriptError with the offending line on bad input."""
    lines = text.splitlines()
    title, hosts, body_start = _parse_front_matter(lines)
    known_hosts = set(hosts)
    turns: list[Turn] = []
    for line_number, line in enumerate(lines[body_start:], start=body_start + 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--"):
            continue
        match = _TURN_RE.match(stripped)
        if match is None:
            raise ScriptError(
                f"script.md line {line_number} is not a turn "
                f"(expected `**Host:** text`): {stripped[:60]!r}"
            )
        speaker, turn_text = match.group(1), match.group(2)
        if speaker not in known_hosts:
            raise ScriptError(
                f"script.md line {line_number} names unknown host {speaker!r} "
                f"(hosts: {', '.join(hosts)})"
            )
        turns.append(Turn(speaker=speaker, text=turn_text))
    return Transcript(title=title, hosts=hosts, turns=turns)
