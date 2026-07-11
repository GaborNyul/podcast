"""Tests for podcast.tts.registry."""

import pytest

from podcast.config import AppConfig, TTSSettings
from podcast.errors import TTSError
from podcast.tts import registry
from podcast.tts.kokoro import KokoroEngine
from podcast.tts.qwen3 import Qwen3Engine


class TestAvailableEngines:
    def test_lists_both_engines(self) -> None:
        assert registry.available_engines() == ["kokoro", "qwen3"]


class TestCreateEngine:
    def test_kokoro(self) -> None:
        config = AppConfig(tts=TTSSettings(engine="kokoro"))
        assert isinstance(registry.create_engine(config), KokoroEngine)

    def test_qwen3_constructs_without_torch(self) -> None:
        config = AppConfig(tts=TTSSettings(engine="qwen3"))
        assert isinstance(registry.create_engine(config), Qwen3Engine)

    def test_unknown_engine_raises_with_available(self) -> None:
        config = AppConfig(tts=TTSSettings(engine="sirens"))
        with pytest.raises(TTSError, match="kokoro, qwen3"):
            registry.create_engine(config)
