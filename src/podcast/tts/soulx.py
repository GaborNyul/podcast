# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""SoulX-Podcast-1.7B: the dialogue-native GPU engine (ADR 0012).

The whole conversation is rendered in one pass, so every line's prosody hears the
lines before it. Voices are zero-shot clones of per-host reference WAVs (the
reference's register drives the emotional baseline); inference code comes from a
commit-pinned checkout of the upstream repo, since no PyPI package exists.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from podcast.config import AppConfig
from podcast.errors import TTSError
from podcast.tts.base import DialogueLine, EngineInfo

REPO_URL = "https://github.com/Soul-AILab/SoulX-Podcast"
REPO_COMMIT = "5ac9c0e1cfe596396200c7d38e3fd53b7b3fbf4b"  # pragma: allowlist secret — git pin
MODEL_ID = "Soul-AILab/SoulX-Podcast-1.7B"
MODEL_REVISION = "7d381e3aa4f2ac9e23201b57957fd0499f8f2fd1"  # pragma: allowlist secret — HF pin
SAMPLE_RATE = 24000
MAX_SPEAKERS = 4
INSTALL_HINT = (
    "the soulx extra is not installed; run `uv sync --extra soulx` "
    "(PyPI deps only — the SoulX source is fetched automatically, pinned to "
    f"{REPO_COMMIT[:12]})"
)
# Delivery-note words that map onto SoulX's documented paralinguistic tags; word-anchored
# so 'insightful' or 'breathtaking' never inject spurious sighs/breaths.
_TAG_PATTERNS = (
    (re.compile(r"\blaugh(s|ing|ter)?\b"), "<|laughter|>"),
    (re.compile(r"\bsigh(s|ing)?\b"), "<|sigh|>"),
    (re.compile(r"\bbreath(s|ing|y)?\b"), "<|breathing|>"),
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


class DialogueModel(Protocol):
    """The slice of soulxpodcast.SoulXPodcast this engine uses."""

    def forward_longform(self, **data: object) -> dict[str, list[Any]]: ...


def ensure_repo(models_dir: Path) -> Path:
    """Commit-pinned checkout of the SoulX inference source under models_dir."""
    repo = models_dir / "soulx" / "SoulX-Podcast"
    if not (repo / "soulxpodcast").is_dir():
        if repo.exists():  # interrupted clone; git refuses non-empty destinations
            shutil.rmtree(repo)
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", REPO_URL, str(repo))
    _git("-C", str(repo), "checkout", "--quiet", REPO_COMMIT)
    return repo


def _git(*args: str) -> None:
    result = subprocess.run(  # noqa: S603 — fixed argv, pinned URL/commit
        ["git", *args],  # noqa: S607 — resolved from PATH like the rest of the toolchain
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise TTSError(f"cannot fetch SoulX source: git {args[0]}: {result.stderr.strip()[:200]}")


def shim_torchaudio() -> None:
    """torchaudio 2.9 nightlies delegate load/save to torchcodec, which is not
    ABI-safe against TheRock wheels; SoulX only needs plain WAV I/O."""
    import soundfile  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]
    import torch  # pyright: ignore[reportMissingImports]
    import torchaudio  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    def _load(path: object, *_args: object, **_kwargs: object) -> tuple[Any, int]:
        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return torch.from_numpy(data.T), int(sr)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]

    def _save(path: object, tensor: Any, sample_rate: int, **_kwargs: object) -> None:
        soundfile.write(str(path), tensor.detach().cpu().numpy().T, sample_rate)  # pyright: ignore[reportUnknownMemberType]

    torchaudio.load = _load
    torchaudio.save = _save


def tagged_text(line: DialogueLine) -> str:
    """One single-line SoulX utterance: paralinguistic tags named by the delivery
    note, then the whitespace-flattened text (upstream's parser is single-line)."""
    text = " ".join(line.text.split())
    if not text:
        raise TTSError(f"speaker {line.speaker!r} has an empty line; SoulX needs spoken text")
    note = line.delivery.lower()
    tags = "".join(tag for pattern, tag in _TAG_PATTERNS if pattern.search(note))
    return f"{tags}{text}"


class SoulXEngine:
    """Whole-conversation synthesis through SoulXPodcast.forward_longform."""

    name = "soulx"

    def __init__(self, config: AppConfig) -> None:
        self._models_dir = config.paths.resolved_models_dir()
        self._refs = dict(config.tts.soulx_refs)
        self._device = config.tts.device or "cuda"
        self._model: DialogueModel | None = None
        self._dataset: object = None
        self._process_single_input: Any = None

    def info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            device=self._device,
            sample_rate=SAMPLE_RATE,
            dialogue_native=True,
            supports_delivery=True,
        )

    def _load(self) -> DialogueModel:
        if self._model is None:
            os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
            os.environ.setdefault("LIBROSA_DISABLE_NUMBA", "1")
            try:
                shim_torchaudio()
                from huggingface_hub import (  # pyright: ignore[reportMissingImports]
                    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
                )
            except ImportError as exc:
                raise TTSError(INSTALL_HINT) from exc
            repo = ensure_repo(self._models_dir)
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            try:
                from cli.podcast import (  # pyright: ignore[reportMissingImports]
                    initiate_model,  # pyright: ignore[reportUnknownVariableType]
                )
                from soulxpodcast.utils.infer_utils import (  # pyright: ignore[reportMissingImports]
                    process_single_input,  # pyright: ignore[reportUnknownVariableType]
                )
            except ImportError as exc:
                raise TTSError(f"SoulX source at {repo} failed to import: {exc}") from exc
            weights = str(snapshot_download(MODEL_ID, revision=MODEL_REVISION))  # pyright: ignore[reportUnknownArgumentType]
            model, dataset = initiate_model(986, weights, "hf", False)  # pyright: ignore[reportUnknownVariableType]
            self._model = cast("DialogueModel", model)
            self._dataset = dataset
            self._process_single_input = process_single_input
        return self._model

    def _reference(self, voice: str) -> tuple[Path, str]:
        ref = self._refs.get(voice)
        if ref is None:
            raise TTSError(f"no SoulX reference for voice {voice!r}; add it under [tts.soulx_refs]")
        wav = Path(ref)
        if not wav.is_absolute() and not wav.is_file() and (_REPO_ROOT / ref).is_file():
            wav = _REPO_ROOT / ref  # the shipped defaults resolve from any cwd
        transcript = wav.with_suffix(".txt")
        if not wav.is_file() or not transcript.is_file():
            raise TTSError(f"SoulX reference {wav} (with sidecar .txt transcript) not found")
        return wav, transcript.read_text(encoding="utf-8").strip()

    def cache_token(self, voice: str) -> str:
        """Identity of the cloned voice: the reference files' content, so re-pointing
        or re-minting a reference invalidates cached dialogue audio."""
        wav, transcript = self._reference(voice)
        digest = hashlib.sha256()
        digest.update(wav.read_bytes())
        digest.update(b"\x00")
        digest.update(transcript.encode("utf-8"))
        return digest.hexdigest()

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        self.synthesize_dialogue(
            [DialogueLine(speaker="solo", text=text, delivery=delivery)],
            {"solo": voice},
            [out_path],
        )

    def synthesize_dialogue(
        self, lines: Sequence[DialogueLine], voices: dict[str, str], out_paths: Sequence[Path]
    ) -> None:
        model = self._load()
        speakers: list[str] = []
        for line in lines:
            if line.speaker not in speakers:
                speakers.append(line.speaker)
        if len(speakers) > MAX_SPEAKERS:
            raise TTSError(f"SoulX supports at most {MAX_SPEAKERS} speakers, got {len(speakers)}")
        refs = [self._reference(voices[speaker]) for speaker in speakers]
        texts = [f"[S{speakers.index(line.speaker) + 1}]{tagged_text(line)}" for line in lines]
        try:
            data = self._process_single_input(
                self._dataset,
                texts,
                [str(wav) for wav, _ in refs],
                [text for _, text in refs],
                False,
                None,
            )
            wavs = model.forward_longform(**data)["generated_wavs"]
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"soulx failed to synthesize dialogue: {exc}") from exc
        if len(wavs) != len(out_paths):
            raise TTSError(f"soulx returned {len(wavs)} turns for {len(out_paths)} lines")
        from podcast.tts.kokoro import write_wav

        # Scratch-then-replace: a crash mid-write must never leave a final path that
        # the whole-dialogue cache would mistake for a completed segment (ADR 0007).
        for wav, out_path in zip(wavs, out_paths, strict=True):
            scratch = out_path.with_suffix(".tmp.wav")
            write_wav(scratch, wav.squeeze(0).cpu().numpy(), SAMPLE_RATE)
            scratch.replace(out_path)
