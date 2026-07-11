"""Tests for podcast.errors."""

import pytest

from podcast import errors


class TestPodcastError:
    def test_is_an_exception(self) -> None:
        assert issubclass(errors.PodcastError, Exception)

    def test_default_exit_code(self) -> None:
        assert errors.PodcastError.exit_code == 1

    def test_message_survives(self) -> None:
        exc = errors.PodcastError("something broke")
        assert str(exc) == "something broke"

    @pytest.mark.parametrize(
        ("exc_type", "expected_code"),
        [
            (errors.ConfigError, 2),
            (errors.IngestError, 3),
            (errors.ProviderError, 4),
            (errors.ScriptError, 5),
            (errors.TTSError, 6),
            (errors.AudioError, 7),
        ],
    )
    def test_subclasses_have_distinct_exit_codes(
        self, exc_type: type[errors.PodcastError], expected_code: int
    ) -> None:
        assert issubclass(exc_type, errors.PodcastError)
        assert exc_type.exit_code == expected_code

    def test_exit_codes_are_unique(self) -> None:
        subclasses = errors.PodcastError.__subclasses__()
        codes = [sub.exit_code for sub in subclasses]
        assert len(codes) == len(set(codes))
