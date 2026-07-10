"""Qwen3-TTS-12Hz-1.7B on ROCm — the GPU engine for the Strix Halo box (ADR 0004).

Hardware notes: gfx1151 requires AMD "TheRock" nightly torch wheels; never set
HSA_OVERRIDE_GFX_VERSION. Inference goes through the official `qwen-tts` package
(`uv sync --extra qwen3` installs both). All heavyweight imports stay inside this
thin adapter; `pytest -m integration -k qwen3` runs the real-model RTF benchmark.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from podcast.config import AppConfig
from podcast.errors import TTSError
from podcast.tts.base import EngineInfo


class SpeechModel(Protocol):
    """The slice of qwen_tts.Qwen3TTSModel this engine uses (typed locally so the
    engine type-checks whether or not the qwen3 extra is installed)."""

    def generate_custom_voice(
        self, *, text: str, language: str, speaker: str
    ) -> tuple[Sequence[object], int]: ...


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAMPLE_RATE = 24000
LANGUAGE = "English"
INSTALL_HINT = (
    "the qwen3 extra is not installed; run `uv sync --extra qwen3` "
    "(pulls the qwen-tts package and TheRock ROCm torch wheels for gfx1151 — see README)"
)


class Qwen3Engine:
    """Per-line synthesis through qwen-tts's Qwen3TTSModel (CustomVoice speakers)."""

    name = "qwen3"

    def __init__(self, config: AppConfig) -> None:
        self._device_override = config.tts.device
        self._model: SpeechModel | None = None
        self._device = self._device_override or "cuda"

    def _load(self) -> SpeechModel:
        if self._model is None:
            try:
                import torch  # pyright: ignore[reportMissingImports]
                from qwen_tts import (  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
                    Qwen3TTSModel,  # pyright: ignore[reportUnknownVariableType]
                )
            except ImportError as exc:
                raise TTSError(INSTALL_HINT) from exc
            device = self._device_override
            if device is None:
                cuda_ok = bool(
                    torch.cuda.is_available()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                )
                device = "cuda" if cuda_ok else "cpu"
            self._device = device
            dtype = (  # pyright: ignore[reportUnknownVariableType]
                torch.bfloat16  # pyright: ignore[reportUnknownMemberType]
                if device.startswith("cuda")
                else torch.float32  # pyright: ignore[reportUnknownMemberType]
            )
            self._model = cast(
                "SpeechModel",
                Qwen3TTSModel.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    MODEL_ID, device_map=device, dtype=dtype
                ),
            )
        return self._model

    def info(self) -> EngineInfo:
        return EngineInfo(name=self.name, device=self._device, sample_rate=SAMPLE_RATE)

    def synthesize_line(self, text: str, voice: str, out_path: Path) -> None:
        model = self._load()
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=text, language=LANGUAGE, speaker=voice
            )
        except Exception as exc:
            raise TTSError(f"qwen3 failed to synthesize (voice {voice!r}): {exc}") from exc
        if not len(wavs):
            raise TTSError("qwen3 returned no audio")
        from podcast.tts.kokoro import write_wav

        write_wav(out_path, wavs[0], int(sample_rate))
