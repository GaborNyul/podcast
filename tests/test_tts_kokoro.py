"""Tests for podcast.tts.kokoro (runtime replaced by a typed double)."""

import sys
import types
import wave
from pathlib import Path
from typing import ClassVar

import httpx
import pytest
import respx

from podcast.errors import TTSError
from podcast.tts import kokoro


class _FakeKokoroRuntime:
    instances: "ClassVar[list[_FakeKokoroRuntime]]" = []
    fail = False

    def __init__(self, model_path: str, voices_path: str) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        _FakeKokoroRuntime.instances.append(self)

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[list[float], int]:
        if self.fail:
            raise RuntimeError("phonemizer exploded")
        del voice, speed, lang
        return [0.0] * max(1, len(text)), kokoro.SAMPLE_RATE


@pytest.fixture
def fake_runtime_module(monkeypatch: pytest.MonkeyPatch) -> type[_FakeKokoroRuntime]:
    module = types.ModuleType("kokoro_onnx")
    module.Kokoro = _FakeKokoroRuntime  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)
    _FakeKokoroRuntime.instances = []
    _FakeKokoroRuntime.fail = False
    return _FakeKokoroRuntime


def _touch_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "kokoro-v1.0.onnx").write_bytes(b"onnx")
    (models_dir / "voices-v1.0.bin").write_bytes(b"voices")


class TestWriteWav:
    def test_writes_16bit_mono(self, tmp_path: Path) -> None:
        out = tmp_path / "x.wav"
        kokoro.write_wav(out, [0.0, 0.5, -0.5, 2.0], 24000)
        with wave.open(str(out), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24000
            assert handle.getnframes() == 4


class TestEnsureModelFiles:
    def test_existing_files_skip_download(self, tmp_path: Path) -> None:
        _touch_models(tmp_path)
        model, voices = kokoro.ensure_model_files(tmp_path)
        assert model.is_file()
        assert voices.is_file()

    def test_downloads_missing_files(self, tmp_path: Path, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(kokoro.MODEL_URL).mock(
            return_value=httpx.Response(200, content=b"onnx-bytes")
        )
        respx_mock.get(kokoro.VOICES_URL).mock(
            return_value=httpx.Response(200, content=b"voice-bytes")
        )
        model, voices = kokoro.ensure_model_files(tmp_path / "models")
        assert model.read_bytes() == b"onnx-bytes"
        assert voices.read_bytes() == b"voice-bytes"
        assert not list((tmp_path / "models").glob("*.part"))

    def test_download_failure_raises_tts_error(
        self, tmp_path: Path, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.get(kokoro.MODEL_URL).mock(return_value=httpx.Response(500))
        with pytest.raises(TTSError, match="cannot download"):
            kokoro.ensure_model_files(tmp_path)


class TestKokoroEngine:
    def test_info(self, tmp_path: Path) -> None:
        engine = kokoro.KokoroEngine(tmp_path)
        info = engine.info()
        assert info.name == "kokoro"
        assert info.device == "cpu"
        assert info.sample_rate == kokoro.SAMPLE_RATE
        assert info.dialogue_native is False
        assert info.supports_delivery is False

    @pytest.mark.usefixtures("fake_runtime_module")
    def test_synthesize_line_writes_wav(self, tmp_path: Path) -> None:
        _touch_models(tmp_path / "models")
        engine = kokoro.KokoroEngine(tmp_path / "models")
        out = tmp_path / "line.wav"
        engine.synthesize_line("hello there", "af_heart", out)
        assert out.is_file()
        with wave.open(str(out), "rb") as handle:
            assert handle.getnframes() == len("hello there")

    @pytest.mark.usefixtures("fake_runtime_module")
    def test_delivery_note_is_ignored(self, tmp_path: Path) -> None:
        _touch_models(tmp_path / "models")
        engine = kokoro.KokoroEngine(tmp_path / "models")
        out = tmp_path / "line.wav"
        engine.synthesize_line("hello", "af_heart", out, delivery="excited, racing ahead")
        assert out.is_file()

    @pytest.mark.usefixtures("fake_runtime_module")
    def test_runtime_loads_once(self, tmp_path: Path) -> None:
        _touch_models(tmp_path / "models")
        engine = kokoro.KokoroEngine(tmp_path / "models")
        engine.synthesize_line("one", "af_heart", tmp_path / "1.wav")
        engine.synthesize_line("two", "af_heart", tmp_path / "2.wav")
        assert len(_FakeKokoroRuntime.instances) == 1

    @pytest.mark.usefixtures("fake_runtime_module")
    def test_runtime_failure_maps_to_tts_error(self, tmp_path: Path) -> None:
        _touch_models(tmp_path / "models")
        engine = kokoro.KokoroEngine(tmp_path / "models")
        _FakeKokoroRuntime.fail = True
        with pytest.raises(TTSError, match="kokoro failed"):
            engine.synthesize_line("boom", "af_heart", tmp_path / "x.wav")

    def test_missing_package_raises_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
        engine = kokoro.KokoroEngine(tmp_path)
        with pytest.raises(TTSError, match="not installed"):
            engine.synthesize_line("x", "af_heart", tmp_path / "x.wav")
