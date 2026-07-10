"""TTS engine protocol (ADR 0003: stitched single-speaker now, dialogue-native later)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EngineInfo:
    """Engine capabilities used for routing and assembly decisions."""

    name: str
    device: str
    sample_rate: int
    dialogue_native: bool = False


@runtime_checkable
class TTSEngine(Protocol):
    """Per-utterance synthesis; dialogue-native engines extend this post-MVP."""

    name: str

    def info(self) -> EngineInfo: ...

    def synthesize_line(self, text: str, voice: str, out_path: Path) -> None:
        """Render one spoken line to a WAV file at `out_path`."""
        ...
