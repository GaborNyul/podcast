"""Tests for podcast.tts.qwen3 (torch/transformers replaced by typed doubles)."""

import sys
import types
import wave
from pathlib import Path

import pytest

from podcast.config import AppConfig, TTSSettings
from podcast.errors import TTSError
from podcast.tts import qwen3


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.result: dict[str, object] = {
            "audio": [0.0, 0.1, -0.1],
            "sampling_rate": 24000,
        }
        self.fail = False

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        if self.fail:
            raise RuntimeError("hip error")
        self.calls.append((text, dict(kwargs)))
        return self.result


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, *, cuda_available: bool
) -> tuple[_FakePipeline, list[dict[str, object]]]:
    pipeline_instance = _FakePipeline()
    created: list[dict[str, object]] = []

    torch_module = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

    torch_module.cuda = _Cuda  # type: ignore[attr-defined]

    transformers_module = types.ModuleType("transformers")

    def fake_pipeline(task: str, **kwargs: object) -> _FakePipeline:
        created.append({"task": task, **kwargs})
        return pipeline_instance

    transformers_module.pipeline = fake_pipeline  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    return pipeline_instance, created


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
            engine.synthesize_line("x", "Ethan", Path("out.wav"))

    def test_synthesize_writes_wav_and_uses_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_instance, created = _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        out = tmp_path / "line.wav"
        engine.synthesize_line("hello", "Ethan", out)
        assert created[0]["task"] == "text-to-audio"
        assert created[0]["model"] == qwen3.MODEL_ID
        assert created[0]["device"] == "cuda"
        assert pipeline_instance.calls[0][1]["forward_params"] == {"speaker": "Ethan"}
        with wave.open(str(out), "rb") as handle:
            assert handle.getnframes() == 3
        assert engine.info().device == "cuda"

    def test_device_override_skips_gpu_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _pipeline, created = _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig(tts=TTSSettings(device="cpu")))
        engine.synthesize_line("hello", "Ethan", tmp_path / "x.wav")
        assert created[0]["device"] == "cpu"
        assert engine.info().device == "cpu"

    def test_falls_back_to_cpu_without_gpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=False)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("hello", "Ethan", tmp_path / "x.wav")
        assert engine.info().device == "cpu"

    def test_pipeline_loads_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _pipeline, created = _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("one", "Ethan", tmp_path / "1.wav")
        engine.synthesize_line("two", "Ethan", tmp_path / "2.wav")
        assert len(created) == 1

    def test_runtime_failure_maps_to_tts_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_instance, _created = _install_fakes(monkeypatch, cuda_available=True)
        pipeline_instance.fail = True
        engine = qwen3.Qwen3Engine(AppConfig())
        with pytest.raises(TTSError, match="qwen3 failed"):
            engine.synthesize_line("boom", "Ethan", tmp_path / "x.wav")

    def test_missing_audio_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pipeline_instance, _created = _install_fakes(monkeypatch, cuda_available=True)
        pipeline_instance.result = {"sampling_rate": 24000}
        engine = qwen3.Qwen3Engine(AppConfig())
        with pytest.raises(TTSError, match="no audio"):
            engine.synthesize_line("x", "Ethan", tmp_path / "x.wav")

    def test_default_sample_rate_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline_instance, _created = _install_fakes(monkeypatch, cuda_available=True)
        pipeline_instance.result = {"audio": [0.0]}
        engine = qwen3.Qwen3Engine(AppConfig())
        out = tmp_path / "x.wav"
        engine.synthesize_line("x", "Ethan", out)
        with wave.open(str(out), "rb") as handle:
            assert handle.getframerate() == qwen3.SAMPLE_RATE
