"""Tests for podcast.cli.app."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

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

    def test_polish_pass_can_be_disabled(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        source = isolated_env / "notes.md"
        source.write_text("# Ants\n\nAnts are strong.", encoding="utf-8")
        (isolated_env / "podcast.toml").write_text(
            '[llm]\nprovider = "fake"\n[script]\npolish_pass = false\n', encoding="utf-8"
        )
        result = runner.invoke(app_mod.app, ["generate", str(source), "-d", "1"])
        assert result.exit_code == 0, result.output
        assert "Polishing dialogue" not in result.output

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

    def __init__(self, *, supports_delivery: bool = False) -> None:
        self.renders = 0
        self.deliveries: list[str] = []
        self.supports_delivery = supports_delivery

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="kokoro",
            device="cpu",
            sample_rate=24000,
            supports_delivery=self.supports_delivery,
        )

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        del text, voice
        self.renders += 1
        self.deliveries.append(delivery)
        out_path.write_bytes(b"RIFF-fake")


class _FakeDialogueEngine:
    name = "soulx"

    def __init__(self) -> None:
        self.dialogue_calls = 0

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="soulx",
            device="cpu",
            sample_rate=24000,
            dialogue_native=True,
            supports_delivery=True,
        )

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        del text, voice, delivery
        out_path.write_bytes(b"RIFF-fake")

    def synthesize_dialogue(self, lines: object, voices: object, out_paths: list[Path]) -> None:
        del lines, voices
        self.dialogue_calls += 1
        for path in out_paths:
            path.write_bytes(b"RIFF-fake")

    def cache_token(self, voice: str) -> str:
        return f"token-{voice}"


def _engine_factory(engine: object) -> Callable[[AppConfig], object]:
    def factory(_config: AppConfig) -> object:
        return engine

    return factory


def _fake_assemble(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def assemble(segment_paths: list[Path], out_path: Path, **kwargs: object) -> None:
        calls.append({"segment_paths": list(segment_paths), **kwargs})
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
            if line.startswith("**Alex"):
                lines[index] = "**Alex:** A brand new hand-edited opening line."
                break
        script.write_text("\n".join(lines), encoding="utf-8")
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.renders == baseline + 1

    def test_assembly_receives_one_gap_scale_per_gap(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        calls = _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        segments = cast("list[Path]", calls[0]["segment_paths"])
        scales = cast("list[float]", calls[0]["gap_scales"])
        assert len(scales) == len(segments) - 1

    def test_delivery_notes_reach_a_supporting_engine(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine(supports_delivery=True)
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        assert "warm, curious" in engine.deliveries  # annotated lines carry their note
        assert "" in engine.deliveries  # neutral lines stay neutral

    def test_host_tempo_derives_faster_segments(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        (isolated_env / "podcast.toml").write_text(
            '[llm]\nprovider = "fake"\n'
            "[[script.hosts]]\n"
            'name = "Alex"\ngender = "male"\npersona = "companion"\n'
            "[[script.hosts]]\n"
            'name = "Maya"\ngender = "female"\npersona = "guide"\ntempo = 1.1\n',
            encoding="utf-8",
        )
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        variants: list[tuple[Path, float]] = []

        def fake_variant(source: Path, tempo: float) -> Path:
            variants.append((source, tempo))
            return source

        monkeypatch.setattr(app_mod, "tempo_variant", fake_variant)
        _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        tempos = {tempo for _, tempo in variants}
        assert tempos == {1.0, 1.1}  # Alex neutral, Maya 10% faster

    def test_dialogue_native_engine_renders_whole_conversation(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = _generate_episode(isolated_env, monkeypatch)
        engine = _FakeDialogueEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))

        def fake_voices(*_args: object) -> dict[str, str]:
            return {"Alex": "alex", "Maya": "maya"}

        monkeypatch.setattr(app_mod, "resolve_voices", fake_voices)
        calls = _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        assert engine.dialogue_calls == 1
        assert "0 cached" in result.output  # stats stay honest on the dialogue path
        segments = cast("list[Path]", calls[0]["segment_paths"])
        assert [path.name for path in segments] == sorted(path.name for path in segments)
        assert len(segments) == 4  # every spoken turn, in script order
        # unchanged script: whole dialogue is a cache hit
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.dialogue_calls == 1
        assert "4 cached, 0 rendered" in result.output
        # any line edit re-renders the whole dialogue (context-dependent prosody)
        script = workspace / "script.md"
        script.write_text(
            script.read_text(encoding="utf-8").replace("the sources", "the papers", 1),
            encoding="utf-8",
        )
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.dialogue_calls == 2

    def test_host_style_is_composed_with_delivery_notes(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        (isolated_env / "podcast.toml").write_text(
            '[llm]\nprovider = "fake"\n'
            "[[script.hosts]]\n"
            'name = "Alex"\ngender = "male"\npersona = "companion"\n'
            'style = "Speak at a fast, energetic pace."\n'
            "[[script.hosts]]\n"
            'name = "Maya"\ngender = "female"\npersona = "guide"\n'
            'style = "Bright and lively."\n',
            encoding="utf-8",
        )
        engine = _FakeEngine(supports_delivery=True)
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        # Alex's lines carry a note -> style composed with it; Maya's are plain -> style alone.
        assert "Speak at a fast, energetic pace.; warm, curious" in engine.deliveries
        assert "Bright and lively." in engine.deliveries

    def test_delivery_notes_are_blanked_for_non_supporting_engine(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0, result.output
        assert engine.renders > 0
        assert set(engine.deliveries) == {""}

    def test_edited_delivery_note_is_free_on_non_supporting_engine(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        runner.invoke(app_mod.app, ["synthesize", "demo"])
        baseline = engine.renders
        script = workspace / "script.md"
        content = script.read_text(encoding="utf-8")
        assert "[warm, curious]" in content
        script.write_text(
            content.replace("[warm, curious]", "[deadpan, slower]", 1), encoding="utf-8"
        )
        result = runner.invoke(app_mod.app, ["synthesize", "demo"])
        assert result.exit_code == 0
        assert engine.renders == baseline  # note is not in the cache key -> all hits

    def test_edited_delivery_note_rerenders_only_one_segment(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = _generate_episode(isolated_env, monkeypatch)
        engine = _FakeEngine(supports_delivery=True)
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        _fake_assemble(monkeypatch)
        runner.invoke(app_mod.app, ["synthesize", "demo"])
        baseline = engine.renders
        script = workspace / "script.md"
        content = script.read_text(encoding="utf-8")
        assert "[warm, curious]" in content
        script.write_text(
            content.replace("[warm, curious]", "[deadpan, slower]", 1), encoding="utf-8"
        )
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


class TestFormatsCommand:
    @pytest.mark.usefixtures("isolated_env")
    def test_lists_all_formats_and_marks_selected(self) -> None:
        result = runner.invoke(app_mod.app, ["formats"])
        assert result.exit_code == 0
        for key in ("deep-dive", "brief", "debate", "critique"):
            assert key in result.output
        assert "deep-dive (selected)" in result.output
        assert "solo" in result.output


class TestFormatSelection:
    def _source(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)
        source = isolated_env / "notes.md"
        source.write_text("# Ants\n\nAnts are impressively strong.", encoding="utf-8")
        return source

    def test_generate_brief_is_solo_and_recorded(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = self._source(isolated_env, monkeypatch)
        result = runner.invoke(
            app_mod.app,
            [
                "generate",
                str(source),
                "--provider",
                "fake",
                "--format",
                "brief",
                "--name",
                "solo-demo",
            ],
        )
        assert result.exit_code == 0, result.output
        script = (isolated_env / "episodes" / "solo-demo" / "script.md").read_text("utf-8")
        assert 'format: "brief"' in script
        assert 'hosts: ["Alex"]' in script
        assert "**Maya" not in script

    def test_unknown_format_fails_with_choices(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = self._source(isolated_env, monkeypatch)
        result = runner.invoke(
            app_mod.app,
            ["generate", str(source), "--provider", "fake", "--format", "sirens"],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ConfigError)
        assert "unknown format" in str(result.exception)

    def test_solo_episode_synthesizes_end_to_end(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = self._source(isolated_env, monkeypatch)
        result = runner.invoke(
            app_mod.app,
            [
                "generate",
                str(source),
                "--provider",
                "fake",
                "--format",
                "brief",
                "--name",
                "solo-demo",
            ],
        )
        assert result.exit_code == 0, result.output
        engine = _FakeEngine()
        monkeypatch.setattr(app_mod, "create_engine", _engine_factory(engine))
        calls = _fake_assemble(monkeypatch)
        result = runner.invoke(app_mod.app, ["synthesize", "solo-demo"])
        assert result.exit_code == 0, result.output
        assert engine.renders > 0
        assert len(calls) == 1

    def test_format_default_minutes_apply_without_explicit_duration(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = self._source(isolated_env, monkeypatch)
        result = runner.invoke(
            app_mod.app,
            ["generate", str(source), "--provider", "fake", "--format", "brief"],
        )
        assert result.exit_code == 0, result.output
        assert "~2.0" in result.output  # brief's 2-minute default, not the config's 10

    def test_explicit_duration_beats_the_format_default(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = self._source(isolated_env, monkeypatch)
        result = runner.invoke(
            app_mod.app,
            ["generate", str(source), "--provider", "fake", "--format", "brief", "-d", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "~1.0" in result.output
