"""Tests for podcast.tts.voices."""

import pytest

from podcast.config import AppConfig, TTSSettings
from podcast.errors import TTSError
from podcast.tts import voices


class TestVoicesFor:
    def test_kokoro_has_both_genders(self) -> None:
        pool = voices.voices_for("kokoro")
        assert {voice.gender for voice in pool} == {"male", "female"}

    def test_qwen3_has_both_genders(self) -> None:
        pool = voices.voices_for("qwen3")
        assert {voice.gender for voice in pool} == {"male", "female"}

    def test_unknown_engine_is_empty(self) -> None:
        assert voices.voices_for("mystery") == []


class TestResolveVoices:
    def test_default_hosts_get_gendered_voices(self) -> None:
        config = AppConfig()
        mapping = voices.resolve_voices(config, "kokoro", ["Alex", "Maya"])
        assert mapping["Alex"] == "am_michael"  # Alex is male in the default config
        assert mapping["Maya"] == "af_heart"

    def test_config_override_wins(self) -> None:
        config = AppConfig(tts=TTSSettings(voices={"Alex": "bm_george"}))
        mapping = voices.resolve_voices(config, "kokoro", ["Alex", "Maya"])
        assert mapping["Alex"] == "bm_george"

    def test_same_gender_hosts_get_distinct_voices(self) -> None:
        config = AppConfig()
        mapping = voices.resolve_voices(config, "kokoro", ["Maya", "Unknown-Female"])
        # Maya is female; the unknown speaker at index 1 defaults to female too.
        assert mapping["Maya"] != mapping["Unknown-Female"]

    def test_unknown_speakers_alternate_genders(self) -> None:
        config = AppConfig()
        mapping = voices.resolve_voices(config, "kokoro", ["Person1", "Person2"])
        pool = {voice.id: voice.gender for voice in voices.voices_for("kokoro")}
        assert pool[mapping["Person1"]] == "male"
        assert pool[mapping["Person2"]] == "female"

    def test_unknown_engine_with_full_overrides_is_fine(self) -> None:
        config = AppConfig(tts=TTSSettings(voices={"Alex": "x", "Maya": "y"}))
        mapping = voices.resolve_voices(config, "mystery", ["Alex", "Maya"])
        assert mapping == {"Alex": "x", "Maya": "y"}

    def test_unknown_engine_without_override_raises(self) -> None:
        config = AppConfig()
        with pytest.raises(TTSError, match="no voice registry"):
            voices.resolve_voices(config, "mystery", ["Alex"])

    def test_soulx_rejects_override_from_another_engine(self) -> None:
        config = AppConfig(tts=TTSSettings(voices={"Alex": "ryan"}))
        with pytest.raises(TTSError, match="'ryan' .* is a qwen3 voice, but the engine is 'soulx'"):
            voices.resolve_voices(config, "soulx", ["Alex", "Maya"])

    def test_soulx_accepts_custom_registered_reference(self) -> None:
        config = AppConfig(
            tts=TTSSettings(
                voices={"Alex": "gabor"},
                soulx_refs={"gabor": "refs/gabor.wav"},
            )
        )
        mapping = voices.resolve_voices(config, "soulx", ["Alex"])
        assert mapping["Alex"] == "gabor"

    def test_soulx_unknown_override_points_at_refs_section(self) -> None:
        config = AppConfig(tts=TTSSettings(voices={"Alex": "ghost"}))
        with pytest.raises(TTSError, match=r"add it under \[tts\.soulx_refs\]"):
            voices.resolve_voices(config, "soulx", ["Alex"])

    def test_kokoro_rejects_override_from_another_engine(self) -> None:
        config = AppConfig(tts=TTSSettings(voices={"Alex": "ryan"}))
        message = "'ryan' .* is a qwen3 voice, but the engine is 'kokoro'"
        with pytest.raises(TTSError, match=message):
            voices.resolve_voices(config, "kokoro", ["Alex"])

    def test_kokoro_allows_voice_outside_curated_registry(self) -> None:
        # The registry lists a curated subset; kokoro itself knows more voices.
        config = AppConfig(tts=TTSSettings(voices={"Alex": "am_liam"}))
        mapping = voices.resolve_voices(config, "kokoro", ["Alex"])
        assert mapping["Alex"] == "am_liam"

    def test_qwen3_override_matches_own_registry_case_insensitively(self) -> None:
        # qwen-tts accepts lowercase speaker names; 'ryan' is qwen3's own Ryan.
        config = AppConfig(tts=TTSSettings(voices={"Alex": "ryan"}))
        mapping = voices.resolve_voices(config, "qwen3", ["Alex"])
        assert mapping["Alex"] == "ryan"

    def test_gender_pool_wraps_around(self) -> None:
        config = AppConfig()
        speakers = [f"M{index}" for index in range(0, 10, 2)]  # five male-position speakers
        mapping = voices.resolve_voices(config, "qwen3", speakers)
        assert len(mapping) == len(speakers)
        assert all(mapping.values())
