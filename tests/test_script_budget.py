"""Tests for podcast.script.budget."""

import pytest

from podcast.config import AppConfig
from podcast.script import budget


class TestCalibrationFor:
    def test_known_engine_uses_config(self) -> None:
        config = AppConfig()
        assert budget.calibration_for(config, "qwen3") == 0.87

    def test_unknown_engine_uses_default(self) -> None:
        config = AppConfig()
        assert budget.calibration_for(config, "mystery") == budget.DEFAULT_CALIBRATION


class TestWordsForMinutes:
    def test_scales_by_wpm_and_calibration(self) -> None:
        assert budget.words_for_minutes(10, 150, 0.85) == 1275

    def test_never_below_one_word(self) -> None:
        assert budget.words_for_minutes(0.001, 150, 0.5) == 1


class TestEpisodeWordBudget:
    def test_uses_config_wpm(self) -> None:
        config = AppConfig()
        assert budget.episode_word_budget(config, 10, "qwen3") == 1305


class TestEstimatedMinutes:
    def test_inverse_of_budget(self) -> None:
        config = AppConfig()
        words = budget.episode_word_budget(config, 15, "qwen3")
        assert budget.estimated_minutes(config, words, "qwen3") == pytest.approx(15, abs=0.01)


class TestWithinTolerance:
    @pytest.mark.parametrize(
        ("actual", "target", "tolerance", "expected"),
        [
            (100, 100, 0.15, True),
            (115, 100, 0.15, True),
            (85, 100, 0.15, True),
            (116, 100, 0.15, False),
            (84, 100, 0.15, False),
            (0, 100, 0.15, False),
        ],
    )
    def test_boundaries(self, actual: int, target: int, tolerance: float, expected: bool) -> None:
        assert budget.within_tolerance(actual, target, tolerance) is expected
