"""Transcript and outline models; tolerant of extra keys from LLM output."""

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """One spoken line, optionally with a performance note for the voice engine."""

    model_config = ConfigDict(extra="ignore")

    speaker: str
    text: str
    delivery: str = Field(
        default="",
        description=(
            "Short performance note for the voice engine (tone, pace, emotional "
            "register — e.g. 'excited, racing ahead'); never spoken; empty for a "
            "neutral read."
        ),
    )


class Transcript(BaseModel):
    """A whole episode script; the unit that round-trips through script.md."""

    model_config = ConfigDict(extra="ignore")

    title: str
    hosts: list[str] = Field(min_length=2)
    turns: list[Turn]

    def word_count(self) -> int:
        return sum(len(turn.text.split()) for turn in self.turns)


class OutlineSegment(BaseModel):
    """One planned episode segment with its word budget."""

    model_config = ConfigDict(extra="ignore")

    heading: str
    target_words: int = Field(ge=1)
    notes: str = ""


class Outline(BaseModel):
    """The word-budgeted episode plan."""

    model_config = ConfigDict(extra="ignore")

    title: str
    segments: list[OutlineSegment] = Field(min_length=1)

    def total_words(self) -> int:
        return sum(segment.target_words for segment in self.segments)


class DialogueChunk(BaseModel):
    """The LLM's per-segment reply; speakers are constrained via a schema enum."""

    model_config = ConfigDict(extra="ignore")

    turns: list[Turn]
