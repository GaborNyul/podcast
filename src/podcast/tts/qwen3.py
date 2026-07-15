"""Qwen3-TTS-12Hz-1.7B on ROCm — the GPU engine for the Strix Halo box (ADR 0004).

Hardware notes: gfx1151 requires AMD "TheRock" nightly torch wheels; never set
HSA_OVERRIDE_GFX_VERSION. Inference goes through the official `qwen-tts` package
(`uv sync --extra qwen3` installs both). All heavyweight imports stay inside this
thin adapter; `pytest -m integration -k qwen3` runs the real-model RTF benchmark.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from podcast import emphasis
from podcast.config import AppConfig
from podcast.errors import TTSError
from podcast.tts.base import EngineInfo


class SpeechModel(Protocol):
    """The slice of qwen_tts.Qwen3TTSModel this engine uses (typed locally so the
    engine type-checks whether or not the qwen3 extra is installed)."""

    def generate_custom_voice(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
    ) -> tuple[Sequence[object], int]: ...


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAMPLE_RATE = 24000
LANGUAGE = "English"
# ~170s of audio at the 12Hz codec — a per-line ceiling so generation can't run away.
MAX_NEW_TOKENS = 2048
INSTALL_HINT = (
    "the qwen3 extra is not installed; run `uv sync --extra qwen3` "
    "(pulls the qwen-tts package and TheRock ROCm torch wheels for gfx1151 — see README)"
)
# ADR 0014: emphasis is best-effort — spans render as CAPS in the spoken text and this
# clause names them in the instruct channel. The 2026-07-15 A/B audition confirmed
# CAPS+instruct as the default strategy; the clause wording lives in this one template.
EMPHASIS_CLAUSE_TEMPLATE = "Put strong emphasis on the {noun} {names}."


# Per-span treatment, tuned by the two-round 2026-07-15 hardware audition:
# CAPS-in-text is the only lever qwen3 reliably follows; the clause merely
# calibrates an actual CAPS change (clause without a CAPS change went 0-for-5,
# stressing the wrong word or nothing). So a span is treated — uppercased in
# the text AND named in the clause — exactly when uppercasing changes a span
# longer than 2 chars. Everything else gets no treatment, only its markup
# stripped: short spans (CAPS read 'it' as the acronym "eye-tee") and spans
# CAPS cannot change (all-caps 'RAG'/'AI', numerals '100').
def _treated(span: str) -> bool:
    """True when the span is uppercased in the text and named in the clause."""
    return len(span) > 2 and span.upper() != span


def _emphasis_clause(span_texts: Sequence[str]) -> str:
    """Instruct clause naming each distinct stressed span as written; empty when none."""
    if not span_texts:
        return ""
    *head, last = (f"'{span}'" for span in dict.fromkeys(span_texts))
    names = f"{', '.join(head)} and {last}" if head else last
    return EMPHASIS_CLAUSE_TEMPLATE.format(noun="words" if head else "word", names=names)


class Qwen3Engine:
    """Per-line synthesis through qwen-tts's Qwen3TTSModel (CustomVoice speakers)."""

    name = "qwen3"

    def __init__(self, config: AppConfig) -> None:
        self._device_override = config.tts.device
        self._model: SpeechModel | None = None
        self._device = self._device_override or "cuda"
        self._temperature = config.tts.qwen3_temperature
        self._top_p = config.tts.qwen3_top_p
        self._repetition_penalty = config.tts.qwen3_repetition_penalty

    def _load(self) -> SpeechModel:
        if self._model is None:
            # New tensor shapes trigger minutes-long exhaustive kernel tuning on
            # gfx1151 unless MIOpen's fast find mode is on (user env wins).
            os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
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
        return EngineInfo(
            name=self.name,
            device=self._device,
            sample_rate=SAMPLE_RATE,
            supports_delivery=True,
            supports_emphasis=True,
        )

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        model = self._load()
        treated = [span for span in emphasis.spans(text) if _treated(span)]
        # With no treated span this stays byte-identical to the unmarked path —
        # the text renders to its markup-stripped form (exactly the text an
        # unmarked script would carry) and the clause is empty, so instruct is
        # the delivery note alone or None.
        clause = _emphasis_clause(treated)
        note = delivery.strip()
        if clause and note.endswith((".", "!", "?")):
            instruct = f"{note} {clause}"
        else:
            instruct = ". ".join(part for part in (note, clause) if part)
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=emphasis.render(text, lambda span: span.upper() if _treated(span) else span),
                language=LANGUAGE,
                speaker=voice,
                instruct=instruct or None,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            raise TTSError(f"qwen3 failed to synthesize (voice {voice!r}): {exc}") from exc
        if not len(wavs):
            raise TTSError("qwen3 returned no audio")
        from podcast.tts.kokoro import write_wav

        write_wav(out_path, wavs[0], int(sample_rate))
