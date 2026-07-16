# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.doctor."""

import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from podcast import doctor
from podcast.config import AppConfig, PathsSettings, TTSSettings


def _config_in(tmp_path: Path) -> AppConfig:
    return AppConfig(
        tts=TTSSettings(engine="kokoro"),
        paths=PathsSettings(episodes_dir=tmp_path / "episodes", models_dir=tmp_path / "models"),
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


class TestCheckKokoro:
    def test_installed_without_models(self, tmp_path: Path) -> None:
        result = doctor.check_kokoro(_config_in(tmp_path))
        assert result.ok
        assert "download on first use" in result.detail

    def test_installed_with_models_is_ready(self, tmp_path: Path) -> None:
        models = tmp_path / "models"
        models.mkdir(parents=True)
        (models / "kokoro-v1.0.onnx").write_bytes(b"x")
        result = doctor.check_kokoro(_config_in(tmp_path))
        assert result.ok
        assert result.detail == "ready"

    def test_missing_package_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        result = doctor.check_kokoro(_config_in(tmp_path))
        assert not result.ok
        assert "uv sync" in result.hint


class TestCheckQwen3:
    def _fake_torch(self, available: bool) -> types.ModuleType:
        torch_module = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return available

            @staticmethod
            def get_device_name(_index: int) -> str:
                return "AMD Radeon Graphics (gfx1151)"

        torch_module.cuda = _Cuda  # type: ignore[attr-defined]
        return torch_module

    def test_hsa_override_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        result = doctor.check_qwen3(_config_in(tmp_path))
        assert not result.ok
        assert "unset it" in result.hint

    def test_missing_torch_hints_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)
        result = doctor.check_qwen3(_config_in(tmp_path))
        assert not result.ok
        assert "--extra qwen3" in result.hint

    def test_no_gpu_visible_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(available=False))
        result = doctor.check_qwen3(_config_in(tmp_path))
        assert not result.ok
        assert "no GPU visible" in result.detail

    def test_gpu_visible_reports_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
        monkeypatch.setitem(sys.modules, "torch", self._fake_torch(available=True))
        result = doctor.check_qwen3(_config_in(tmp_path))
        assert result.ok
        assert "gfx1151" in result.detail


class TestCheckEngine:
    def test_dispatches_to_kokoro(self, tmp_path: Path) -> None:
        config = _config_in(tmp_path)
        config.tts.engine = "kokoro"
        assert doctor.check_engine(config).name == "kokoro engine"

    def test_dispatches_to_qwen3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        config = _config_in(tmp_path)
        config.tts.engine = "qwen3"
        assert doctor.check_engine(config).name == "qwen3 engine"

    def test_dispatches_to_soulx_missing_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "s3tokenizer", None)
        config = _config_in(tmp_path)
        config.tts.engine = "soulx"
        result = doctor.check_engine(config)
        assert result.name == "soulx engine"
        assert not result.ok
        assert "uv sync --extra soulx" in result.hint

    def test_soulx_missing_reference_fails_with_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        monkeypatch.setitem(sys.modules, "s3tokenizer", types.ModuleType("s3tokenizer"))
        config = _config_in(tmp_path)
        config.tts.engine = "soulx"
        config.tts.soulx_refs = {"alex": str(tmp_path / "gone.wav")}
        result = doctor.check_engine(config)
        assert not result.ok
        assert "reference" in result.hint

    def test_soulx_with_shipped_refs_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        monkeypatch.setitem(sys.modules, "s3tokenizer", types.ModuleType("s3tokenizer"))
        config = _config_in(tmp_path)
        config.tts.engine = "soulx"
        result = doctor.check_engine(config)
        assert result.ok, result.detail

    def test_unknown_engine_fails(self, tmp_path: Path) -> None:
        config = _config_in(tmp_path)
        config.tts.engine = "sirens"
        result = doctor.check_engine(config)
        assert not result.ok
        assert "unknown engine" in result.detail


class TestRunChecks:
    def test_runs_all_checks_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.doctor.shutil.which", _which_found)
        monkeypatch.setattr("podcast.doctor.subprocess.run", _run_factory(0, "ffmpeg version 7.1"))
        results = doctor.run_checks(_config_in(tmp_path))
        assert [result.name for result in results] == [
            "ffmpeg",
            "episodes dir",
            "models dir",
            "kokoro engine",
        ]
        assert all(result.ok for result in results)
