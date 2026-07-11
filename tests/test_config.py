"""Tests for podcast.config."""

import os
from pathlib import Path

import pytest

from podcast import config as config_mod
from podcast.config import (
    AppConfig,
    HostSpec,
    PathsSettings,
    load_config,
    project_config_path,
    user_config_path,
)
from podcast.errors import ConfigError


class TestLoadConfig:
    @pytest.mark.usefixtures("isolated_env")
    def test_defaults_with_no_files(self) -> None:
        config = load_config()
        assert config.llm.provider == "ollama"
        assert config.script.words_per_minute == 150
        assert config.tts.engine == "qwen3"
        assert len(config.script.hosts) == 2
        assert {host.gender for host in config.script.hosts} == {"male", "female"}

    def test_user_file_applies(self, isolated_env: Path) -> None:
        user_file = isolated_env / "user.toml"
        user_file.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
        config = load_config(user_file=user_file)
        assert config.llm.provider == "openai"

    def test_project_file_overrides_user_file(self, isolated_env: Path) -> None:
        user_file = isolated_env / "user.toml"
        user_file.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
        project_file = isolated_env / "podcast.toml"
        project_file.write_text('[llm]\nprovider = "anthropic"\n', encoding="utf-8")
        config = load_config(project_file=project_file, user_file=user_file)
        assert config.llm.provider == "anthropic"

    def test_layers_deep_merge_within_sections(self, isolated_env: Path) -> None:
        user_file = isolated_env / "user.toml"
        user_file.write_text('[llm]\nmodel = "llama3:70b"\n', encoding="utf-8")
        project_file = isolated_env / "podcast.toml"
        project_file.write_text('[llm]\nprovider = "ollama-cloud"\n', encoding="utf-8")
        config = load_config(project_file=project_file, user_file=user_file)
        assert config.llm.model == "llama3:70b"
        assert config.llm.provider == "ollama-cloud"

    def test_scalar_layers_replace_not_merge(self, isolated_env: Path) -> None:
        user_file = isolated_env / "user.toml"
        user_file.write_text("[script]\ndefault_minutes = 20\n", encoding="utf-8")
        project_file = isolated_env / "podcast.toml"
        project_file.write_text("[script]\ndefault_minutes = 5\n", encoding="utf-8")
        config = load_config(project_file=project_file, user_file=user_file)
        assert config.script.default_minutes == 5

    def test_env_overrides_files(self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        project_file = isolated_env / "podcast.toml"
        project_file.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
        monkeypatch.setenv("PODCAST_LLM__PROVIDER", "anthropic")
        config = load_config(project_file=project_file)
        assert config.llm.provider == "anthropic"

    def test_project_file_discovered_in_cwd(self, isolated_env: Path) -> None:
        (isolated_env / "podcast.toml").write_text('[tts]\nengine = "kokoro"\n', encoding="utf-8")
        config = load_config()
        assert config.tts.engine == "kokoro"

    def test_invalid_toml_raises_config_error(self, isolated_env: Path) -> None:
        broken = isolated_env / "podcast.toml"
        broken.write_text("[llm\nprovider=", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(project_file=broken)

    def test_unreadable_file_raises_config_error(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = isolated_env / "podcast.toml"
        target.write_text("", encoding="utf-8")

        def boom(*_args: object, **_kwargs: object) -> object:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "open", boom)
        with pytest.raises(ConfigError, match="cannot read config file"):
            load_config(project_file=target)

    def test_wrong_value_type_raises_config_error(self, isolated_env: Path) -> None:
        bad = isolated_env / "podcast.toml"
        bad.write_text('[script]\nwords_per_minute = "many"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid configuration"):
            load_config(project_file=bad)

    def test_too_few_hosts_raises_config_error(self, isolated_env: Path) -> None:
        bad = isolated_env / "podcast.toml"
        bad.write_text(
            '[[script.hosts]]\nname = "Solo"\ngender = "male"\npersona = "alone"\n',
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="invalid configuration"):
            load_config(project_file=bad)

    def test_unknown_keys_are_ignored(self, isolated_env: Path) -> None:
        extra = isolated_env / "podcast.toml"
        extra.write_text("[future_section]\nsomething = 1\n", encoding="utf-8")
        config = load_config(project_file=extra)
        assert config.llm.provider == "ollama"

    def test_unicode_values_survive(self, isolated_env: Path) -> None:
        unicode_file = isolated_env / "podcast.toml"
        unicode_file.write_text(
            '[[script.hosts]]\nname = "Ági"\ngender = "female"\npersona = "műsorvezető 🎙️"\n'
            '[[script.hosts]]\nname = "Ödön"\ngender = "male"\npersona = "szakértő"\n',
            encoding="utf-8",
        )
        config = load_config(project_file=unicode_file)
        assert config.script.hosts[0].name == "Ági"
        assert config.script.hosts[1].persona == "szakértő"


class TestDotenvSupport:
    def test_dotenv_file_feeds_settings_and_process_env(self, isolated_env: Path) -> None:
        (isolated_env / ".env").write_text("PODCAST_LLM__PROVIDER=fake\n", encoding="utf-8")
        try:
            config = load_config()
            assert config.llm.provider == "fake"
            assert os.environ["PODCAST_LLM__PROVIDER"] == "fake"
        finally:
            os.environ.pop("PODCAST_LLM__PROVIDER", None)

    def test_real_environment_wins_over_dotenv(
        self, isolated_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (isolated_env / ".env").write_text("PODCAST_LLM__PROVIDER=fake\n", encoding="utf-8")
        monkeypatch.setenv("PODCAST_LLM__PROVIDER", "anthropic")
        config = load_config()
        assert config.llm.provider == "anthropic"


class TestUserConfigPath:
    def test_honors_xdg_config_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/somewhere/config")
        assert user_config_path() == Path("/somewhere/config/podcast/config.toml")

    def test_defaults_to_home_dot_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert user_config_path() == tmp_path / ".config" / "podcast" / "config.toml"


class TestProjectConfigPath:
    def test_is_cwd_relative(self) -> None:
        assert project_config_path() == Path("podcast.toml")


class TestResolvedModelsDir:
    def test_explicit_dir_wins(self) -> None:
        paths = PathsSettings(models_dir=Path("/opt/models"))
        assert paths.resolved_models_dir() == Path("/opt/models")

    def test_defaults_to_xdg_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        paths = PathsSettings()
        assert paths.resolved_models_dir() == tmp_path / "podcast" / "models"

    def test_defaults_to_home_cache_without_xdg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        paths = PathsSettings()
        assert paths.resolved_models_dir() == tmp_path / ".cache" / "podcast" / "models"


class TestAppConfig:
    @pytest.mark.usefixtures("isolated_env")
    def test_nested_env_delimiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PODCAST_LLM__API_KEY", "sk-test-123")  # pragma: allowlist secret
        config = AppConfig()
        assert config.llm.api_key == "sk-test-123"  # pragma: allowlist secret

    @pytest.mark.usefixtures("isolated_env")
    def test_default_calibration_covers_both_engines(self) -> None:
        config = AppConfig()
        assert set(config.tts.calibration) == {"qwen3", "kokoro"}

    @pytest.mark.usefixtures("isolated_env")
    def test_expressiveness_defaults(self) -> None:
        config = AppConfig()
        assert config.script.polish_pass is True
        assert config.tts.qwen3_temperature == 0.8
        assert config.tts.qwen3_top_p == 0.9
        assert config.tts.qwen3_repetition_penalty == 1.05

    def test_host_spec_rejects_unknown_gender(self) -> None:
        with pytest.raises(ValueError, match="gender"):
            HostSpec(name="X", gender="robot", persona="beep")  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("name", ["Alex [AI]", "co]host", "DJ: Live"])
    def test_host_spec_rejects_script_grammar_characters(self, name: str) -> None:
        with pytest.raises(ValueError, match="may not contain"):
            HostSpec(name=name, gender="male", persona="p")

    def test_host_spec_rejects_blank_name(self) -> None:
        with pytest.raises(ValueError, match="must not be blank"):
            HostSpec(name="   ", gender="male", persona="p")

    def test_host_tempo_defaults_neutral_and_is_bounded(self) -> None:
        assert HostSpec(name="A", gender="male", persona="p").tempo == 1.0
        with pytest.raises(ValueError, match="tempo"):
            HostSpec(name="A", gender="male", persona="p", tempo=2.5)

    @pytest.mark.usefixtures("isolated_env")
    def test_format_defaults_to_deep_dive(self) -> None:
        config = AppConfig()
        assert config.script.format == "deep-dive"
        assert config.script.solo_host is None

    def test_unknown_format_is_rejected_with_choices(self) -> None:
        with pytest.raises(ValueError, match="deep-dive, brief, debate, critique"):
            config_mod.ScriptSettings(format="sirens")

    @pytest.mark.parametrize("key", ["deep-dive", "brief", "debate", "critique"])
    def test_known_formats_are_accepted(self, key: str) -> None:
        assert config_mod.ScriptSettings(format=key).format == key

    def test_solo_host_must_be_a_configured_host(self) -> None:
        with pytest.raises(ValueError, match="not a configured host"):
            config_mod.ScriptSettings(solo_host="Zed")

    def test_solo_host_accepts_a_configured_host(self) -> None:
        assert config_mod.ScriptSettings(solo_host="Maya").solo_host == "Maya"

    def test_module_exposes_no_mutable_singleton(self) -> None:
        assert not hasattr(config_mod, "CONFIG")
