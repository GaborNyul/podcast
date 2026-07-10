"""Qwen3-TTS-12Hz-1.7B on ROCm — the GPU engine for the Strix Halo box (ADR 0004).

Hardware notes: gfx1151 requires AMD "TheRock" nightly torch wheels (installed via
`uv sync --extra qwen3`); never set HSA_OVERRIDE_GFX_VERSION. All torch-touching
code stays inside this thin adapter; `pytest -m integration` runs the real-model
RTF benchmark on the target machine.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

from podcast.config import AppConfig
from podcast.errors import TTSError
from podcast.tts.base import EngineInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    type SpeechPipeline = Callable[..., Mapping[str, object]]

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAMPLE_RATE = 24000
INSTALL_HINT = (
    "the qwen3 extra is not installed; run `uv sync --extra qwen3` "
    "(pulls TheRock ROCm torch wheels for gfx1151 — see README)"
)


class Qwen3Engine:
    """Per-line synthesis through a transformers text-to-speech pipeline."""

    name = "qwen3"

    def __init__(self, config: AppConfig) -> None:
        self._device_override = config.tts.device
        self._pipeline: SpeechPipeline | None = None
        self._device = self._device_override or "cuda"

    def _load(self) -> "SpeechPipeline":
        if self._pipeline is None:
            try:
                import torch  # pyright: ignore[reportMissingImports]
                from transformers import (  # pyright: ignore[reportMissingImports]
                    pipeline,  # pyright: ignore[reportUnknownVariableType]
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
            self._pipeline = cast(
                "SpeechPipeline",
                pipeline(
                    "text-to-audio",
                    model=MODEL_ID,
                    device=device,
                    trust_remote_code=True,
                ),
            )
        return self._pipeline

    def info(self) -> EngineInfo:
        return EngineInfo(name=self.name, device=self._device, sample_rate=SAMPLE_RATE)

    def synthesize_line(self, text: str, voice: str, out_path: Path) -> None:
        speech = self._load()
        try:
            result = speech(text, forward_params={"speaker": voice})
        except Exception as exc:
            raise TTSError(f"qwen3 failed to synthesize (voice {voice!r}): {exc}") from exc
        audio = result.get("audio")
        rate = result.get("sampling_rate", SAMPLE_RATE)
        if audio is None:
            raise TTSError("qwen3 pipeline returned no audio")
        from podcast.tts.kokoro import write_wav

        write_wav(out_path, audio, int(cast("int", rate)))
