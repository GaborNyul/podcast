"""Content-aware pause scaling between turns (ADR 0011).

Pause structure is a real affect channel: backchannels land fast, interruptions
faster, question hand-offs stay snappy, and an ellipsis earns a beat. Scales
multiply the sampled inter-turn silence in `podcast.audio.assemble`.
"""

from collections.abc import Sequence

from podcast.script.models import Turn

_BACKCHANNEL_MAX_WORDS = 4

INTERRUPT_SCALE = 0.4
BACKCHANNEL_SCALE = 0.5
HANDOFF_SCALE = 0.7
BEAT_SCALE = 1.4


def _scale(previous: Turn, following: Turn) -> float:
    text = previous.text.rstrip()
    if text.endswith(("—", "-")):
        return INTERRUPT_SCALE
    if len(following.text.split()) <= _BACKCHANNEL_MAX_WORDS:
        return BACKCHANNEL_SCALE
    if text.endswith("?"):
        return HANDOFF_SCALE
    if text.endswith(("...", "…")):
        return BEAT_SCALE
    return 1.0


def gap_scales(turns: Sequence[Turn]) -> list[float]:
    """One pause scale per gap between consecutive spoken turns."""
    return [
        _scale(previous, following) for previous, following in zip(turns, turns[1:], strict=False)
    ]
