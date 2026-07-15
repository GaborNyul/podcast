"""TTS engine protocol (ADR 0003: stitched single-speaker now, dialogue-native later)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DialogueLine:
    """One turn handed to a dialogue-native engine."""

    speaker: str
    text: str
    delivery: str = ""


@dataclass(frozen=True)
class EngineInfo:
    """Engine capabilities used for routing and assembly decisions."""

    name: str
    device: str
    sample_rate: int
    dialogue_native: bool = False
    supports_delivery: bool = False
    supports_emphasis: bool = False


@runtime_checkable
class TTSEngine(Protocol):
    """Per-utterance synthesis; dialogue-native engines extend this post-MVP."""

    name: str

    def info(self) -> EngineInfo: ...

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        """Render one spoken line to a WAV file at `out_path`.

        `delivery` is a short performance note (tone, pace, emotional register);
        engines that cannot act on it declare `supports_delivery=False` and ignore it.
        `text` carries `*word*` emphasis markup (ADR 0014) only when the engine
        declares `supports_emphasis=True`; engines that declare `False` always
        receive markup-free text (the CLI strips it), so they need no handling.
        """
        ...


@runtime_checkable
class DialogueEngine(Protocol):
    """Whole-conversation synthesis (`dialogue_native=True` in EngineInfo)."""

    name: str

    def info(self) -> EngineInfo: ...

    def synthesize_dialogue(
        self, lines: Sequence[DialogueLine], voices: dict[str, str], out_paths: Sequence[Path]
    ) -> None:
        """Render the whole conversation, one WAV per line at `out_paths[i]`;
        prosody on every line may depend on all preceding lines.

        Each line's `text` carries `*word*` emphasis markup (ADR 0014) only when
        the engine declares `supports_emphasis=True`; engines that declare `False`
        always receive markup-free text (the CLI strips it), so they need no handling.
        """
        ...

    def cache_token(self, voice: str) -> str:
        """Content identity of what this voice id resolves to (e.g. clone-reference
        bytes); joins the dialogue cache key so voice redefinitions re-render."""
        ...
