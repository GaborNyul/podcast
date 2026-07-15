"""Word-level emphasis markup `*word*` shared by the script and TTS layers (ADR 0014)."""

import re
from collections.abc import Callable

# One valid emphasis span: single asterisks around a non-empty run with no `*`
# inside and no leading/trailing whitespace (internal whitespace is allowed).
EMPHASIS_RE = re.compile(r"\*([^*\s](?:[^*]*[^*\s])?)\*")

_FRAGMENT_CONTEXT = 12


def spans(text: str) -> list[str]:
    """Return the emphasis span texts in document order."""
    return [match.group(1) for match in EMPHASIS_RE.finditer(text)]


def strip_markup(text: str) -> str:
    """Remove markup, keep words: unwrapping valid spans and dropping strays is
    exactly removing every `*`, so the output never contains one."""
    return text.replace("*", "")


def normalize(text: str) -> str:
    """Tolerant cleanup for LLM output: keep valid spans, drop stray/malformed `*`."""
    return _render(text, lambda span: f"*{span}*")


def render_caps(text: str) -> str:
    """Render each valid span as its text uppercased; stray `*` are dropped."""
    return _render(text, str.upper)


def validate(text: str) -> None:
    """Raise ValueError naming the offending fragment if any `*` is not part of a valid span."""
    last = 0
    for match in EMPHASIS_RE.finditer(text):
        _reject_stray(text, last, match.start())
        last = match.end()
    _reject_stray(text, last, len(text))


def _render(text: str, render_span: Callable[[str], str]) -> str:
    """Rebuild text with valid spans passed through `render_span` and stray `*` dropped."""
    out: list[str] = []
    last = 0
    for match in EMPHASIS_RE.finditer(text):
        out.append(text[last : match.start()].replace("*", ""))
        out.append(render_span(match.group(1)))
        last = match.end()
    out.append(text[last:].replace("*", ""))
    return "".join(out)


def _reject_stray(text: str, start: int, end: int) -> None:
    """Raise if text[start:end] — a region between valid spans — contains a `*`."""
    offset = text.find("*", start, end)
    if offset == -1:
        return
    fragment = text[max(0, offset - _FRAGMENT_CONTEXT) : offset + _FRAGMENT_CONTEXT]
    raise ValueError(f"stray or malformed emphasis '*' near {fragment!r}")
