"""TTS engine registry; heavy runtimes import lazily inside their factories."""

from podcast.config import AppConfig
from podcast.errors import TTSError
from podcast.tts.base import TTSEngine

ENGINE_NAMES = ("kokoro", "qwen3", "soulx")


def available_engines() -> list[str]:
    return list(ENGINE_NAMES)


def create_engine(config: AppConfig) -> TTSEngine:
    """Build the configured engine; unknown names fail with the available list."""
    name = config.tts.engine
    if name == "kokoro":
        from podcast.tts.kokoro import KokoroEngine

        return KokoroEngine(config.paths.resolved_models_dir())
    if name == "qwen3":
        from podcast.tts.qwen3 import Qwen3Engine

        return Qwen3Engine(config)
    if name == "soulx":
        from podcast.tts.soulx import SoulXEngine

        return SoulXEngine(config)
    raise TTSError(f"unknown TTS engine {name!r} (available: {', '.join(ENGINE_NAMES)})")
