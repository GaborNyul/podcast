# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Length control: minutes → target words, adjusted by per-engine calibration."""

from podcast.config import AppConfig

DEFAULT_CALIBRATION = 0.85


def calibration_for(config: AppConfig, engine: str) -> float:
    """Measured words-per-minute correction for `engine` (see ADR 0006)."""
    return config.tts.calibration.get(engine, DEFAULT_CALIBRATION)


def words_for_minutes(minutes: float, words_per_minute: int, calibration: float) -> int:
    return max(1, round(minutes * words_per_minute * calibration))


def episode_word_budget(config: AppConfig, minutes: float, engine: str) -> int:
    return words_for_minutes(
        minutes, config.script.words_per_minute, calibration_for(config, engine)
    )


def estimated_minutes(config: AppConfig, words: int, engine: str) -> float:
    """Expected rendered duration for a script of `words` words."""
    effective_wpm = config.script.words_per_minute * calibration_for(config, engine)
    return words / effective_wpm


def within_tolerance(actual_words: int, target_words: int, tolerance: float) -> bool:
    return abs(actual_words - target_words) <= tolerance * target_words
