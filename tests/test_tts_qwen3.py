"""Tests for podcast.tts.qwen3 (torch/qwen-tts replaced by typed doubles)."""

import sys
import types
import wave
from pathlib import Path
from typing import ClassVar

import pytest

from podcast.config import AppConfig, TTSSettings
from podcast.errors import TTSError
from podcast.tts import qwen3


class _FakeModel:
    from_pretrained_calls: "ClassVar[list[dict[str, object]]]" = []
    generate_calls: "ClassVar[list[dict[str, object]]]" = []
    fail = False
    empty = False

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> "_FakeModel":
        cls.from_pretrained_calls.append({"model_id": model_id, **kwargs})
        return cls()

    def generate_custom_voice(
        self, *, text: str, language: str, speaker: str
    ) -> tuple[list[list[float]], int]:
        if _FakeModel.fail:
            raise RuntimeError("hip error")
        _FakeModel.generate_calls.append({"text": text, "language": language, "speaker": speaker})
        if _FakeModel.empty:
            return [], 24000
        return [[0.0, 0.1, -0.1]], 24000


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool) -> None:
    _FakeModel.from_pretrained_calls = []
    _FakeModel.generate_calls = []
    _FakeModel.fail = False
    _FakeModel.empty = False

    torch_module = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    torch_module.cuda = _Cuda  # type: ignore[attr-defined]
    torch_module.bfloat16 = "bf16-sentinel"  # type: ignore[attr-defined]
    torch_module.float32 = "fp32-sentinel"  # type: ignore[attr-defined]

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSModel = _FakeModel  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)


class TestQwen3Engine:
    def test_info_before_load_uses_declared_device(self) -> None:
        engine = qwen3.Qwen3Engine(AppConfig())
        assert engine.info().device == "cuda"
        assert engine.info().sample_rate == qwen3.SAMPLE_RATE

    def test_device_override_from_config(self) -> None:
        config = AppConfig(tts=TTSSettings(device="cpu"))
        assert qwen3.Qwen3Engine(config).info().device == "cpu"

    def test_missing_extra_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        engine = qwen3.Qwen3Engine(AppConfig())
        with pytest.raises(TTSError, match="uv sync --extra qwen3"):
            engine.synthesize_line("x", "Ryan", Path("out.wav"))

    def test_synthesize_writes_wav_on_gpu_with_bf16(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        out = tmp_path / "line.wav"
        engine.synthesize_line("hello", "Ryan", out)
        load = _FakeModel.from_pretrained_calls[0]
        assert load["model_id"] == qwen3.MODEL_ID
        assert load["device_map"] == "cuda"
        assert load["dtype"] == "bf16-sentinel"
        assert _FakeModel.generate_calls[0] == {
            "text": "hello",
            "language": "English",
            "speaker": "Ryan",
        }
        with wave.open(str(out), "rb") as handle:
            assert handle.getnframes() == 3
        assert engine.info().device == "cuda"

    def test_device_override_skips_gpu_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig(tts=TTSSettings(device="cpu")))
        engine.synthesize_line("hello", "Ryan", tmp_path / "x.wav")
        load = _FakeModel.from_pretrained_calls[0]
        assert load["device_map"] == "cpu"
        assert load["dtype"] == "fp32-sentinel"
        assert engine.info().device == "cpu"

    def test_falls_back_to_cpu_without_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=False)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("hello", "Ryan", tmp_path / "x.wav")
        assert engine.info().device == "cpu"

    def test_model_loads_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("one", "Ryan", tmp_path / "1.wav")
        engine.synthesize_line("two", "Ryan", tmp_path / "2.wav")
        assert len(_FakeModel.from_pretrained_calls) == 1

    def test_runtime_failure_maps_to_tts_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        _FakeModel.fail = True
        engine = qwen3.Qwen3Engine(AppConfig())
        with pytest.raises(TTSError, match="qwen3 failed"):
            engine.synthesize_line("boom", "Ryan", tmp_path / "x.wav")

    def test_empty_audio_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        _FakeModel.empty = True
        engine = qwen3.Qwen3Engine(AppConfig())
        with pytest.raises(TTSError, match="no audio"):
            engine.synthesize_line("x", "Ryan", tmp_path / "x.wav")
