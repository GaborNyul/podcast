# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""ffmpeg assembly: concat + randomized inter-turn silence + EBU R128 loudnorm (ADR 0008)."""

import random
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from podcast.errors import AudioError

LOUDNORM_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"
_SILENCE_STEP_MS = 50  # silence durations round to this so gap files can be reused


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise AudioError("ffmpeg not found on PATH; install it (e.g. sudo apt install ffmpeg)")
    return path


def _run(command: Sequence[str]) -> None:
    # S603: fixed argv; the binary comes from shutil.which and paths from our workspace.
    result = subprocess.run(list(command), capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-3:]
        raise AudioError(f"ffmpeg failed: {' | '.join(tail) or 'unknown error'}")


def _silence_file(ffmpeg: str, work_dir: Path, duration_ms: int, sample_rate: int) -> Path:
    path = work_dir / f"silence-{duration_ms}ms.wav"
    if not path.is_file():
        _run(
            [
                ffmpeg,
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={sample_rate}:cl=mono",
                "-t",
                f"{duration_ms / 1000:.3f}",
                "-sample_fmt",
                "s16",
                str(path),
            ]
        )
    return path


def tempo_variant(source: Path, tempo: float) -> Path:
    """Pitch-preserving tempo-adjusted sibling of a rendered segment.

    Derived files live next to the source (`<stem>-tempo<pct>.wav`) and are
    reused, so tempo changes never re-run the synthesis engine.
    """
    if tempo == 1.0:
        return source
    target = source.with_name(f"{source.stem}-tempo{round(tempo * 100)}.wav")
    if target.is_file():
        return target
    ffmpeg = find_ffmpeg()
    scratch = target.with_suffix(".tmp.wav")
    _run([ffmpeg, "-y", "-i", str(source), "-filter:a", f"atempo={tempo}", str(scratch)])
    scratch.replace(target)
    return target


def _pause_ms(rng: random.Random, minimum_ms: int, maximum_ms: int, scale: float = 1.0) -> int:
    raw = rng.randint(minimum_ms, maximum_ms) if maximum_ms > minimum_ms else minimum_ms
    return max(_SILENCE_STEP_MS, round(raw * scale / _SILENCE_STEP_MS) * _SILENCE_STEP_MS)


def _concat_entry(path: Path) -> str:
    """One concat-demuxer line; inside single quotes ffmpeg needs ' written as '\\''."""
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def assemble_episode(
    segment_paths: Sequence[Path],
    out_path: Path,
    *,
    work_dir: Path,
    sample_rate: int,
    pause_min_ms: int,
    pause_max_ms: int,
    bitrate: str,
    seed: int | None = None,
    gap_scales: Sequence[float] | None = None,
) -> None:
    """Concatenate segments with natural pauses, loudness-normalize, export MP3.

    `gap_scales` (one per gap, from `podcast.audio.pacing`) multiplies each
    sampled pause so the silences follow the conversation's rhythm.
    """
    if not segment_paths:
        raise AudioError("no audio segments to assemble")
    if gap_scales is not None and len(gap_scales) != len(segment_paths) - 1:
        raise AudioError(
            f"gap_scales has {len(gap_scales)} entries for "
            f"{len(segment_paths) - 1} gaps between segments"
        )
    ffmpeg = find_ffmpeg()
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)  # noqa: S311 — pause jitter, not cryptography

    entries: list[Path] = []
    for index, segment in enumerate(segment_paths):
        if index > 0:
            scale = gap_scales[index - 1] if gap_scales is not None else 1.0
            duration = _pause_ms(rng, pause_min_ms, pause_max_ms, scale)
            entries.append(_silence_file(ffmpeg, work_dir, duration, sample_rate))
        entries.append(segment)

    concat_list = work_dir / "concat.txt"
    concat_list.write_text("".join(_concat_entry(path) for path in entries), encoding="utf-8")
    combined = work_dir / "combined.wav"
    _run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-ar",
            str(sample_rate),
            str(combined),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(combined),
            "-af",
            LOUDNORM_FILTER,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(out_path),
        ]
    )
