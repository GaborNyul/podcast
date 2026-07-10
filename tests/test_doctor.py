"""Tests for podcast.doctor."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from podcast import doctor
from podcast.config import AppConfig, PathsSettings


def _config_in(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsSettings(episodes_dir=tmp_path / "episodes", models_dir=tmp_path / "models")
    )


def _completed(returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["ffmpeg"], returncode=returncode, stdout=stdout)


def _which_found(_name: str) -> str:
    return "/usr/bin/ffmpeg"


def _which_missing(_name: str) -> None:
    return None


def _run_factory(returncode: int, stdout: str) -> "Callable[..., subprocess.CompletedProcess[str]]":
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(returncode, stdout)

    return fake_run


class TestCheckFfmpeg:
    def test_reports_version_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_found)
        monkeypatch.setattr(
            "podcast.doctor.subprocess.run",
            _run_factory(0, "ffmpeg version 7.1\nbuilt with gcc"),
        )
        result = doctor.check_ffmpeg()
        assert result.ok
        assert result.detail == "ffmpeg version 7.1"

    def test_missing_binary_fails_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_missing)
        result = doctor.check_ffmpeg()
        assert not result.ok
        assert "install ffmpeg" in result.hint

    def test_broken_binary_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_found)
        monkeypatch.setattr("podcast.doctor.subprocess.run", _run_factory(1, ""))
        result = doctor.check_ffmpeg()
        assert not result.ok
        assert "failed to run" in result.detail

    def test_empty_version_output_is_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_found)
        monkeypatch.setattr("podcast.doctor.subprocess.run", _run_factory(0, ""))
        result = doctor.check_ffmpeg()
        assert result.ok
        assert result.detail == "unknown version"


class TestCheckEpisodesDir:
    def test_creates_and_probes_directory(self, tmp_path: Path) -> None:
        config = _config_in(tmp_path)
        result = doctor.check_episodes_dir(config)
        assert result.ok
        assert (tmp_path / "episodes").is_dir()
        assert not (tmp_path / "episodes" / ".doctor-probe").exists()

    def test_unwritable_directory_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config_in(tmp_path)

        def deny(*_args: object, **_kwargs: object) -> None:
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "mkdir", deny)
        result = doctor.check_episodes_dir(config)
        assert not result.ok
        assert "not writable" in result.detail
        assert "episodes_dir" in result.hint


class TestCheckModelsDir:
    def test_creates_models_directory(self, tmp_path: Path) -> None:
        config = _config_in(tmp_path)
        result = doctor.check_models_dir(config)
        assert result.ok
        assert (tmp_path / "models").is_dir()

    def test_uncreatable_directory_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config_in(tmp_path)

        def deny(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "mkdir", deny)
        result = doctor.check_models_dir(config)
        assert not result.ok
        assert "models_dir" in result.hint


class TestRunChecks:
    def test_runs_all_checks_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_found)
        monkeypatch.setattr("podcast.doctor.subprocess.run", _run_factory(0, "ffmpeg version 7.1"))
        results = doctor.run_checks(_config_in(tmp_path))
        assert [result.name for result in results] == ["ffmpeg", "episodes dir", "models dir"]
        assert all(result.ok for result in results)
