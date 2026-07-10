"""Tests for podcast.cli.app."""

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from podcast import __version__
from podcast.cli import app as app_mod
from podcast.config import AppConfig
from podcast.doctor import CheckResult
from podcast.errors import ConfigError

runner = CliRunner()


def _fake_checks(
    results: list[CheckResult],
) -> Callable[[AppConfig], list[CheckResult]]:
    def run_checks(_config: AppConfig) -> list[CheckResult]:
        return results

    return run_checks


class TestMainOptions:
    def test_version_flag(self) -> None:
        result = runner.invoke(app_mod.app, ["--version"])
        assert result.exit_code == 0
        assert f"podcast {__version__}" in result.output

    def test_no_arguments_shows_help(self) -> None:
        result = runner.invoke(app_mod.app, [])
        assert result.exit_code == 0
        assert "Usage" in result.output


class TestDoctorCommand:
    @pytest.mark.usefixtures("isolated_env")
    def test_exit_zero_when_all_checks_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "podcast.cli.app.doctor.run_checks",
            _fake_checks([CheckResult(name="ffmpeg", ok=True, detail="version 7.1")]),
        )
        result = runner.invoke(app_mod.app, ["doctor"])
        assert result.exit_code == 0
        assert "ffmpeg" in result.output

    @pytest.mark.usefixtures("isolated_env")
    def test_exit_one_when_a_check_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "podcast.cli.app.doctor.run_checks",
            _fake_checks(
                [CheckResult(name="ffmpeg", ok=False, detail="missing", hint="install it")]
            ),
        )
        result = runner.invoke(app_mod.app, ["doctor"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestConfigCommand:
    @pytest.mark.usefixtures("isolated_env")
    def test_prints_resolved_config_as_json(self) -> None:
        result = runner.invoke(app_mod.app, ["config"])
        assert result.exit_code == 0
        assert '"provider"' in result.output
        assert "ollama" in result.output

    def test_config_error_is_not_swallowed(self, isolated_env: Path) -> None:
        (isolated_env / "podcast.toml").write_text("[broken", encoding="utf-8")
        result = runner.invoke(app_mod.app, ["config"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ConfigError)


class TestMain:
    def test_renders_podcast_error_with_exit_code(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def explode() -> None:
            raise ConfigError("bad config")

        monkeypatch.setattr(app_mod, "app", explode)
        with pytest.raises(SystemExit) as excinfo:
            app_mod.main()
        assert excinfo.value.code == 2
        assert "bad config" in capsys.readouterr().err

    def test_passes_through_clean_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(app_mod, "app", lambda: None)
        app_mod.main()
