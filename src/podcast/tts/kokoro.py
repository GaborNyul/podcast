"""Kokoro-82M via kokoro-onnx: CPU-only, torch-free, the works-everywhere engine."""

import wave
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from podcast.errors import TTSError
from podcast.tts.base import EngineInfo

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

SAMPLE_RATE = 24000
MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=600.0) as client,
            client.stream("GET", url) as response,
        ):
            response.raise_for_status()
            with scratch.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        raise TTSError(f"cannot download Kokoro model file {url}: {exc}") from exc
    scratch.replace(destination)


def ensure_model_files(models_dir: Path) -> tuple[Path, Path]:
    """Paths of the Kokoro ONNX weights and voice bank, downloading once (~310 MB)."""
    model_path = models_dir / "kokoro-v1.0.onnx"
    voices_path = models_dir / "voices-v1.0.bin"
    if not model_path.is_file():
        _download(MODEL_URL, model_path)
    if not voices_path.is_file():
        _download(VOICES_URL, voices_path)
    return model_path, voices_path


def write_wav(path: Path, samples: object, sample_rate: int) -> None:
    """Write mono float32 samples in [-1, 1] as 16-bit PCM."""
    import numpy

    array = numpy.asarray(samples, dtype=numpy.float32)
    pcm = (numpy.clip(array, -1.0, 1.0) * 32767.0).astype(numpy.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class KokoroEngine:
    """Per-line synthesis through the kokoro-onnx runtime."""

    name = "kokoro"

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._runtime: Kokoro | None = None

    def _load(self) -> "Kokoro":
        if self._runtime is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise TTSError(f"kokoro-onnx is not installed: {exc}") from exc
            model_path, voices_path = ensure_model_files(self._models_dir)
            self._runtime = Kokoro(str(model_path), str(voices_path))
        return self._runtime

    def info(self) -> EngineInfo:
        return EngineInfo(name=self.name, device="cpu", sample_rate=SAMPLE_RATE)

    def synthesize_line(self, text: str, voice: str, out_path: Path, *, delivery: str = "") -> None:
        del delivery  # Kokoro-82M has no prosody input; notes are silently ignored
        runtime = self._load()
        try:
            samples, sample_rate = runtime.create(text, voice=voice, speed=1.0, lang="en-us")
        except Exception as exc:
            raise TTSError(f"kokoro failed to synthesize (voice {voice!r}): {exc}") from exc
        write_wav(out_path, samples, int(sample_rate))
