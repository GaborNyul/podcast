"""Voice registry and speaker→voice resolution (gender defaults, config overrides)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from podcast.config import AppConfig
from podcast.errors import TTSError


@dataclass(frozen=True)
class Voice:
    """One engine voice and the host gender it defaults to."""

    id: str
    gender: Literal["male", "female"]


ENGINE_VOICES: dict[str, list[Voice]] = {
    "kokoro": [
        Voice("af_heart", "female"),
        Voice("am_michael", "male"),
        Voice("af_bella", "female"),
        Voice("am_adam", "male"),
        Voice("bf_emma", "female"),
        Voice("bm_george", "male"),
    ],
    # Qwen3-TTS speaker names; adjust in [tts.voices] if the model card revises them.
    "qwen3": [
        Voice("Cherry", "female"),
        Voice("Ethan", "male"),
        Voice("Serena", "female"),
        Voice("Chelsie", "female"),
    ],
}


def voices_for(engine: str) -> list[Voice]:
    return list(ENGINE_VOICES.get(engine, []))


def resolve_voices(config: AppConfig, engine: str, speakers: Sequence[str]) -> dict[str, str]:
    """Map each speaker to a voice id: config override first, then gender defaults."""
    available = voices_for(engine)
    gender_of = {host.name: host.gender for host in config.script.hosts}
    pools: dict[str, list[Voice]] = {
        "male": [voice for voice in available if voice.gender == "male"],
        "female": [voice for voice in available if voice.gender == "female"],
    }
    counters = {"male": 0, "female": 0}
    mapping: dict[str, str] = {}
    for index, speaker in enumerate(speakers):
        override = config.tts.voices.get(speaker)
        if override is not None:
            mapping[speaker] = override
            continue
        if not available:
            raise TTSError(
                f"no voice registry for engine {engine!r}; set [tts.voices] "
                f"for speaker {speaker!r} in podcast.toml"
            )
        gender = gender_of.get(speaker, "male" if index % 2 == 0 else "female")
        pool = pools[gender] or available
        mapping[speaker] = pool[counters[gender] % len(pool)].id
        counters[gender] += 1
    return mapping
