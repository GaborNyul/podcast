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
    # Qwen3-TTS-12Hz CustomVoice speakers (model card); every speaker can speak
    # English — Ryan/Aiden are the English-native voices.
    "qwen3": [
        Voice("Serena", "female"),
        Voice("Ryan", "male"),
        Voice("Vivian", "female"),
        Voice("Aiden", "male"),
        Voice("Sohee", "female"),
    ],
    # SoulX voices are clone references registered in [tts.soulx_refs].
    "soulx": [
        Voice("maya", "female"),
        Voice("alex", "male"),
    ],
}


def voices_for(engine: str) -> list[Voice]:
    return list(ENGINE_VOICES.get(engine, []))


def _check_override(config: AppConfig, engine: str, speaker: str, voice: str) -> None:
    """Reject a [tts.voices] override that cannot work on the active engine.

    soulx is closed-set (a voice must be a [tts.soulx_refs] key); the other
    registries are curated subsets, so their engines only reject an override
    that belongs to a different engine's registry.
    """
    if engine == "soulx":
        if voice in config.tts.soulx_refs:
            return
    elif any(known.id.lower() == voice.lower() for known in voices_for(engine)):
        return  # qwen-tts accepts speaker names case-insensitively
    foreign = next(
        (
            other
            for other, pool in ENGINE_VOICES.items()
            if other != engine and any(known.id.lower() == voice.lower() for known in pool)
        ),
        None,
    )
    if foreign is not None:
        raise TTSError(
            f"voice {voice!r} for speaker {speaker!r} is a {foreign} voice, but the "
            f"engine is {engine!r}; update [tts.voices] in podcast.toml"
        )
    if engine == "soulx":
        raise TTSError(
            f"no SoulX reference for voice {voice!r} (speaker {speaker!r}); "
            f"add it under [tts.soulx_refs]"
        )


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
            _check_override(config, engine, speaker, override)
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
