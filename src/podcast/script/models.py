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
    # A solo format (brief) produces a one-host transcript; config still
    # requires two configured hosts, the episode just uses one of them.
    hosts: list[str] = Field(min_length=1)
    turns: list[Turn]
    # Which audio-overview format wrote this script (provenance; synthesis
    # does not depend on it).
    format: str = "deep-dive"

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
    host_angles: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Debate format only: the stance each host argues for the whole "
            "episode, keyed by host name; empty for every other format."
        ),
    )

    def total_words(self) -> int:
        return sum(segment.target_words for segment in self.segments)


class DialogueChunk(BaseModel):
    """The LLM's per-segment reply; speakers are constrained via a schema enum."""

    model_config = ConfigDict(extra="ignore")

    turns: list[Turn]


class ReviewFinding(BaseModel):
    """One critique finding, anchored to the material and paired with a fix."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(description="The weakness, gap, or assumption in one sentence.")
    anchor: str = Field(
        description=(
            "What the document actually says that this finding points at — a short "
            "quote or close paraphrase. A finding without an anchor must be dropped."
        )
    )
    detail: str = Field(description="Why this matters to the document's goal.")
    suggestion: str = Field(description="A concrete, actionable fix.")


class CritiqueReview(BaseModel):
    """The critique format's structured pre-outline review of the source material."""

    model_config = ConfigDict(extra="ignore")

    strengths: list[str] = Field(
        min_length=1, description="What genuinely works, each specific to this document."
    )
    findings: list[ReviewFinding] = Field(
        min_length=3, description="The weaknesses that survive re-checking, most important first."
    )
    questions: list[str] = Field(
        default_factory=list,
        description="Open questions the document raises but does not answer.",
    )
