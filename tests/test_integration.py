"""Real-model integration tests (nightly CI / the Strix Halo box; never the default run).

The qwen3 test doubles as the RTF benchmark from the design spec: run it on the
target machine with `uv run pytest -m integration -k qwen3 -s` and read the
printed realtime factor and rendered words-per-minute, then record the wpm-based
calibration in [tts.calibration].
"""

import importlib.util
import shutil
import time
import wave
from pathlib import Path

import pytest

from podcast.config import AppConfig, PathsSettings, TTSSettings

pytestmark = pytest.mark.integration

BENCH_TEXT = (
    "Solar sails use the gentle pressure of sunlight to move spacecraft without any "
    "fuel at all. Over weeks of steady acceleration they can reach speeds that "
    "chemical rockets cannot touch, and missions like IKAROS have already proven "
    "the idea works in practice."
)


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


class TestKokoroRealModel:
    def test_synthesize_and_assemble(self, tmp_path: Path) -> None:
        from podcast.audio.assemble import assemble_episode
        from podcast.tts.kokoro import KokoroEngine

        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed")
        config = AppConfig(paths=PathsSettings(models_dir=tmp_path / "models"))
        engine = KokoroEngine(config.paths.resolved_models_dir())
        first = tmp_path / "a.wav"
        second = tmp_path / "b.wav"
        engine.synthesize_line("Welcome to the show.", "am_michael", first)
        engine.synthesize_line("Great to be here.", "af_heart", second)
        assert _wav_seconds(first) > 0.4
        episode = tmp_path / "episode.mp3"
        assemble_episode(
            [first, second],
            episode,
            work_dir=tmp_path / "work",
            sample_rate=engine.info().sample_rate,
            pause_min_ms=200,
            pause_max_ms=400,
            bitrate="128k",
            seed=1,
        )
        assert episode.stat().st_size > 10_000


class TestQwen3Benchmark:
    def test_rtf_and_calibration(self, tmp_path: Path) -> None:
        if importlib.util.find_spec("torch") is None:
            pytest.skip("qwen3 extra not installed (uv sync --extra qwen3)")
        import torch  # pyright: ignore[reportMissingImports]

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            pytest.skip("no GPU visible to torch")
        from podcast.tts.qwen3 import Qwen3Engine

        engine = Qwen3Engine(AppConfig(tts=TTSSettings(engine="qwen3")))
        out = tmp_path / "bench.wav"
        started = time.monotonic()
        engine.synthesize_line(BENCH_TEXT, "Ethan", out)
        elapsed = time.monotonic() - started
        audio_seconds = _wav_seconds(out)
        words = len(BENCH_TEXT.split())
        rtf = elapsed / audio_seconds
        rendered_wpm = words / (audio_seconds / 60)
        print(f"\nqwen3 RTF: {rtf:.2f} (synth {elapsed:.1f}s for {audio_seconds:.1f}s audio)")
        print(f"qwen3 rendered wpm: {rendered_wpm:.0f}")
        # words_for_minutes(m, wpm, cal) = m*wpm*cal must equal m*rendered_wpm
        print(f"suggested [tts.calibration] qwen3 = {rendered_wpm / 150:.2f}")
        assert audio_seconds > 5.0
