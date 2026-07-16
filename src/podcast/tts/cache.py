# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Content-addressed per-line segment cache: edit one line, re-render one line (ADR 0007)."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheStats:
    """Hit/miss counters surfaced to the user after synthesis."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses


def segment_key(engine: str, voice: str, text: str, delivery: str = "") -> str:
    """Stable identity of one rendered line; any input change changes the key.

    `delivery` only carries a value for engines that act on it, so engines that
    ignore performance notes keep one cache entry per line regardless of notes.
    """
    digest = hashlib.sha256()
    for part in (engine, voice, text, delivery):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def segment_path(segments_dir: Path, key: str) -> Path:
    return segments_dir / f"{key}.wav"


def ensure_segment(
    segments_dir: Path,
    engine: str,
    voice: str,
    text: str,
    delivery: str,
    render: Callable[[Path], None],
    stats: CacheStats,
) -> Path:
    """Return the cached WAV for this line, rendering it (atomically) on a miss."""
    path = segment_path(segments_dir, segment_key(engine, voice, text, delivery))
    if path.is_file():
        stats.hits += 1
        return path
    segments_dir.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".tmp.wav")
    render(scratch)
    scratch.replace(path)
    stats.misses += 1
    return path
