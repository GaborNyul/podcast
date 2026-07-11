"""Transcript ⇄ script.md, the lossless contract between generate and synthesize.

Format: a front-matter block with JSON-encoded values, then exactly one line per
turn (`**Host:** text`, or `**Host [delivery note]:** text` when the turn carries
a performance note). Turn text and delivery are whitespace-normalized on write so
the round-trip is byte-stable and hand edits stay unambiguous. Host names may not
contain the grammar's own characters (`[`, `]`, `:`); podcast.config enforces the
same rule, and the parser rejects front matter that violates it.
"""

import json
import re

from podcast.errors import ScriptError
from podcast.script.models import Transcript, Turn

_EDIT_HINT = (
    "<!-- Edit freely: one line per turn, formatted `**Host:** text` or "
    "`**Host [delivery note]:** text`. Keep host names from the list above; "
    "then run `podcast synthesize`. -->"
)
_TURN_RE = re.compile(r"^\*\*(.+?):\*\* ?(.*)$")
_DELIVERY_RE = re.compile(r"^(.*?)\s*\[(.*)\]$")
_DELIVERY_UNSAFE = re.compile(r"[\[\]:]")


def normalize_turn_text(text: str) -> str:
    """Collapse all whitespace so every turn serializes to a single line."""
    return " ".join(text.split())


def normalize_delivery(text: str) -> str:
    """Whitespace-collapse a delivery note and drop the characters that would
    break the `**Host [note]:** text` line grammar."""
    return " ".join(_DELIVERY_UNSAFE.sub(" ", text).split())


def turn_to_line(turn: Turn) -> str:
    """One `**Host:** text` line; a non-empty delivery note rides in the speaker token."""
    delivery = normalize_delivery(turn.delivery)
    speaker = f"{turn.speaker} [{delivery}]" if delivery else turn.speaker
    return f"**{speaker}:** {normalize_turn_text(turn.text)}"


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
        lines.append(turn_to_line(turn))
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
            for host in hosts:
                if _DELIVERY_UNSAFE.search(host):
                    raise ScriptError(
                        f"front matter host {host!r} may not contain '[', ']' or ':' "
                        "(they break the `**Host [note]:** text` line grammar)"
                    )
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


def _split_speaker(token: str, known_hosts: set[str]) -> tuple[str, str] | None:
    """Resolve a speaker token to (host, delivery). Host names cannot contain
    brackets (validated in the front matter), so the split is unambiguous."""
    if token in known_hosts:
        return token, ""
    match = _DELIVERY_RE.match(token)
    if match is not None and match.group(1) in known_hosts:
        return match.group(1), normalize_delivery(match.group(2))
    return None


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
        token, turn_text = match.group(1), match.group(2)
        resolved = _split_speaker(token, known_hosts)
        if resolved is None:
            raise ScriptError(
                f"script.md line {line_number} names unknown host {token!r} "
                f"(hosts: {', '.join(hosts)})"
            )
        speaker, delivery = resolved
        turns.append(Turn(speaker=speaker, text=turn_text, delivery=delivery))
    return Transcript(title=title, hosts=hosts, turns=turns)
