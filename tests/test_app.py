"""Tests for podcast.cli.app."""

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from podcast import __version__
from podcast.cli import app as app_mod
from podcast.config import AppConfig
from podcast.doctor import CheckResult
from podcast.errors import ConfigError, ScriptError
from podcast.tts.base import EngineInfo

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


class TestGenerateCommand:
    def test_fake_provider_end_to_end(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        source = isolated_env / "notes.md"
        source.write_text("# Ants\n\nAnts can carry fifty times their weight.", encoding="utf-8")
        (isolated_env / "podcast.toml").write_text('[llm]\nprovider = "fake"\n', encoding="utf-8")
        result = runner.invoke(app_mod.app, ["generate", str(source), "-d", "1"])
        assert result.exit_code == 0, result.output
        roots = list((isolated_env / "episodes").iterdir())
        assert len(roots) == 1
        workspace = roots[0]
        assert (workspace / "script.md").is_file()
        assert (workspace / "transcript.json").is_file()
        assert (workspace / "outline.json").is_file()
        assert (workspace / "sources.json").is_file()
        assert "**" in (workspace / "script.md").read_text(encoding="utf-8")

    def test_name_override_sets_slug(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        source = isolated_env / "notes.txt"
        source.write_text("Ants are strong.", encoding="utf-8")
        result = runner.invoke(
            app_mod.app,
            [
                "generate",
                str(source),
                "-d",
                "1",
                "--provider",
                "fake",
                "--engine",
                "kokoro",
                "--name",
                "my-slug",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (isolated_env / "episodes" / "my-slug" / "script.md").is_file()

    def test_traversal_name_is_rejected(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        source = isolated_env / "notes.txt"
        source.write_text("Ants are strong.", encoding="utf-8")
        result = runner.invoke(
            app_mod.app,
            ["generate", str(source), "-d", "1", "--provider", "fake", "--name", "../../escaped"],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ScriptError)
        assert not (isolated_env / "episodes").exists()
        assert not (isolated_env.parent / "escaped").exists()

    def test_missing_source_fails_with_ingest_error(self, isolated_env: Path) -> None:
        result = runner.invoke(
            app_mod.app,
            ["generate", str(isolated_env / "ghost.md"), "--provider", "fake"],
        )
        assert result.exit_code != 0


class _FakeEngine:
    name = "kokoro"

    def __init__(self) -> None:
        self.renders = 0

    def info(self) -> EngineInfo:
        return EngineInfo(name="kokoro", device="cpu", sample_rate=24000)

    def synthesize_line(self, text: str, voice: str, out_path: Path) -> None:
        del text, voice
        self.renders += 1
        out_path.write_bytes(b"RIFF-fake")


def _engine_factory(engine: "_FakeEngine") -> Callable[[AppConfig], "_FakeEngine"]:
    def factory(_config: AppConfig) -> _FakeEngine:
        return engine

    return factory


def _fake_assemble(monkeypatch: pytest.MonkeyPatch) -> list[list[Path]]:
    calls: list[list[Path]] = []

    def assemble(segment_paths: list[Path], out_path: Path, **_kwargs: object) -> None:
        calls.append(list(segment_paths))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ID3-fake-mp3")

    monkeypatch.setattr(app_mod, "assemble_episode", assemble)
    return calls


def _generate_episode(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
    source = isolated_env / "notes.md"
    source.write_text("# Ants\n\nAnts are impressively strong.", encoding="utf-8")
    result = runner.invoke(
        app_mod.app,
        ["generate", str(source), "-d", "1", "--provider", "fake", "--name", "demo"],
    )
    assert result.exit_code == 0, result.output
    return isolated_env / "episodes" / "demo"


class TestSynthesizeCommand:
    def test_renders_all_lines_and_assembles(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        calls = _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        assert engine.renders > 0
        assert len(calls) == 1
        assert (isolated_env / "episodes" / "demo" / "episode.mp3").is_file()

    def test_second_run_hits_cache(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        runner.invoke(app_mod.app, ["synthesize", "demo"])
        first_renders = engine.renders
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.renders == first_renders  # everything cached

    def test_edited_line_rerenders_only_one_segment(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        runner.invoke(app_mod.app, ["synthesize", "demo"])
        baseline = engine.renders
        script = workspace / "script.md"
        lines = script.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("**Alex:**"):
                lines[index] = "**Alex:** A brand new hand-edited opening line."
                break
        script.write_text("\n".join(lines), encoding="utf-8")
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.renders == baseline + 1

    def test_defaults_to_most_recent_episode(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize"])
        assert result.exit_code == 0, result.output

    def test_no_episodes_fails_clearly(self, isolated_env: Path) -> None:
        del isolated_env
        result = runner.invoke(app_mod.app, ["synthesize"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ScriptError)


class TestCreateCommand:
    def test_generates_and_synthesizes_in_one_run(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        source = isolated_env / "notes.txt"
        source.write_text("Ants lift many times their body weight.", encoding="utf-8")
        result = runner.invoke(
            app_mod.app,
            ["create", str(source), "-d", "1", "--provider", "fake", "--name", "one-shot"],
        )
        assert result.exit_code == 0, result.output
        workspace = isolated_env / "episodes" / "one-shot"
        assert (workspace / "script.md").is_file()
        assert (workspace / "episode.mp3").is_file()
        assert engine.renders > 0


class TestEnginesCommand:
    @pytest.mark.usefixtures("isolated_env")
    def test_lists_both_engines_with_status(self) -> None:
        result = runner.invoke(app_mod.app, ["engines"])
        assert result.exit_code == 0
        assert "kokoro" in result.output
        assert "qwen3" in result.output
        assert "selected" in result.output


class TestVoicesCommand:
    @pytest.mark.usefixtures("isolated_env")
    def test_lists_voices_and_mapping(self) -> None:
        result = runner.invoke(app_mod.app, ["voices", "--engine", "kokoro"])
        assert result.exit_code == 0
        assert "af_heart" in result.output
        assert "Alex -> am_michael" in result.output

    @pytest.mark.usefixtures("isolated_env")
    def test_defaults_to_configured_engine(self) -> None:
        result = runner.invoke(app_mod.app, ["voices"])
        assert result.exit_code == 0
        assert "qwen3 voices" in result.output


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
