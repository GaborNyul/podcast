"""Layered configuration: defaults < user TOML < project TOML < environment."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from podcast.errors import ConfigError


class HostSpec(BaseModel):
    """One podcast host: display name, voice-selection gender, persona brief."""

    name: str
    gender: Literal["male", "female"]
    persona: str


def _default_hosts() -> list[HostSpec]:
    return [
        HostSpec(
            name="Alex",
            gender="male",
            persona=(
                "curious generalist host who steers the conversation, asks sharp "
                "questions, and keeps things grounded for newcomers"
            ),
        ),
        HostSpec(
            name="Maya",
            gender="female",
            persona=(
                "expert co-host who explains the material with vivid analogies "
                "and concrete examples"
            ),
        ),
    ]


class LLMSettings(BaseModel):
    """Script-LLM provider selection and sampling parameters."""

    provider: str = "ollama"
    model: str | None = None  # None → the provider's default model (see llm.registry)
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 300.0
    max_retries: int = 2
    context_window: int = 262144
    outline_temperature: float = 0.3
    dialogue_temperature: float = 0.8


class ScriptSettings(BaseModel):
    """Length control and host personas."""

    words_per_minute: int = 150
    default_minutes: int = 10
    length_tolerance: float = 0.15
    hosts: list[HostSpec] = Field(default_factory=_default_hosts, min_length=2)


class TTSSettings(BaseModel):
    """Engine selection, voice mapping, and per-engine duration calibration."""

    engine: str = "qwen3"
    device: str | None = None
    voices: dict[str, str] = Field(default_factory=dict)
    calibration: dict[str, float] = Field(default_factory=lambda: {"qwen3": 0.85, "kokoro": 0.85})


class AudioSettings(BaseModel):
    """Assembly parameters; `seed` pins pause randomization for reproducible builds."""

    pause_min_ms: int = 200
    pause_max_ms: int = 1000
    mp3_bitrate: str = "192k"
    seed: int | None = None


class PathsSettings(BaseModel):
    """Filesystem locations; `models_dir=None` means the XDG cache default."""

    episodes_dir: Path = Path("episodes")
    models_dir: Path | None = None

    def resolved_models_dir(self) -> Path:
        if self.models_dir is not None:
            return self.models_dir
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        return cache_home / "podcast" / "models"


class AppConfig(BaseSettings):
    """Resolved application configuration; env vars use PODCAST_SECTION__KEY."""

    model_config = SettingsConfigDict(
        env_prefix="PODCAST_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    script: ScriptSettings = Field(default_factory=ScriptSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003 — fixed pydantic-settings hook signature
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # init kwargs carry the merged TOML layers; env must still win over them.
        return (env_settings, init_settings)


def user_config_path() -> Path:
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return config_home / "podcast" / "config.toml"


def project_config_path() -> Path:
    return Path("podcast.toml")


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(
                cast("Mapping[str, object]", existing),
                cast("Mapping[str, object]", value),
            )
        else:
            merged[key] = value
    return merged


def load_config(project_file: Path | None = None, user_file: Path | None = None) -> AppConfig:
    """Build the layered AppConfig; raises ConfigError on unreadable/invalid input."""
    user_path = user_file if user_file is not None else user_config_path()
    project_path = project_file if project_file is not None else project_config_path()

    merged: dict[str, object] = {}
    for path in (user_path, project_path):
        if path.is_file():
            merged = _deep_merge(merged, _read_toml(path))

    try:
        # BaseSettings.__init__ types its private `_case_sensitive`-style kwargs, so a
        # dynamic **dict[str, object] cannot be expressed without widening to Any.
        return AppConfig(**merged)  # type: ignore[arg-type]
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc
