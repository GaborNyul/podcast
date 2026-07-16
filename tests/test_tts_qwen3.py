# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.tts.qwen3 (torch/qwen-tts replaced by typed doubles)."""

import os
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
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None = None,
        temperature: float = 0.9,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 0,
    ) -> tuple[list[list[float]], int]:
        if _FakeModel.fail:
            raise RuntimeError("hip error")
        _FakeModel.generate_calls.append(
            {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
            }
        )
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
        assert engine.info().supports_delivery is True
        assert engine.info().supports_emphasis is True

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
            "instruct": None,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_new_tokens": qwen3.MAX_NEW_TOKENS,
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

    def test_delivery_note_is_forwarded_as_instruct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "hello", "Ryan", tmp_path / "x.wav", delivery="excited, racing ahead"
        )
        assert _FakeModel.generate_calls[0]["instruct"] == "excited, racing ahead"

    def test_load_defaults_miopen_fast_find_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
        qwen3.Qwen3Engine(AppConfig()).synthesize_line("x", "Ryan", tmp_path / "x.wav")
        assert os.environ["MIOPEN_FIND_MODE"] == "FAST"

    def test_load_respects_existing_miopen_find_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        monkeypatch.setenv("MIOPEN_FIND_MODE", "NORMAL")
        qwen3.Qwen3Engine(AppConfig()).synthesize_line("x", "Ryan", tmp_path / "x.wav")
        assert os.environ["MIOPEN_FIND_MODE"] == "NORMAL"

    def test_sampling_overrides_come_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        config = AppConfig(
            tts=TTSSettings(qwen3_temperature=0.6, qwen3_top_p=0.8, qwen3_repetition_penalty=1.2)
        )
        engine = qwen3.Qwen3Engine(config)
        engine.synthesize_line("hello", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["temperature"] == 0.6
        assert call["top_p"] == 0.8
        assert call["repetition_penalty"] == 1.2

    def test_blank_delivery_note_becomes_no_instruct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("hello", "Ryan", tmp_path / "x.wav", delivery="   ")
        assert _FakeModel.generate_calls[0]["instruct"] is None

    def test_emphasis_renders_caps_and_instruct_clause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("That's the *whole* point", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "That's the WHOLE point"
        assert call["instruct"] == "Put strong emphasis on the word 'whole'."

    def test_emphasis_clause_appends_to_delivery_note(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's the *whole* point",
            "Ryan",
            tmp_path / "x.wav",
            delivery="excited, racing ahead",
        )
        assert _FakeModel.generate_calls[0]["instruct"] == (
            "excited, racing ahead. Put strong emphasis on the word 'whole'."
        )

    def test_emphasis_clause_after_terminal_punctuated_delivery_adds_no_extra_period(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's the *whole* point",
            "Ryan",
            tmp_path / "x.wav",
            delivery="Speak at a fast, energetic pace.",
        )
        assert _FakeModel.generate_calls[0]["instruct"] == (
            "Speak at a fast, energetic pace. Put strong emphasis on the word 'whole'."
        )

    def test_emphasis_clause_after_question_mark_delivery_keeps_single_separator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's the *whole* point", "Ryan", tmp_path / "x.wav", delivery="Ready?"
        )
        assert _FakeModel.generate_calls[0]["instruct"] == (
            "Ready? Put strong emphasis on the word 'whole'."
        )

    def test_emphasis_clause_after_ellipsis_delivery_adds_no_extra_period(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Any non-alphanumeric terminal already separates — '…' must not
        # collect a bolted-on '.' the way the old '.'/'!'/'?' allowlist did.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's the *whole* point", "Ryan", tmp_path / "x.wav", delivery="trailing off…"
        )
        assert _FakeModel.generate_calls[0]["instruct"] == (
            "trailing off… Put strong emphasis on the word 'whole'."
        )

    def test_emphasis_clause_names_repeated_span_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("*big*, really *big*", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "BIG, really BIG"
        assert call["instruct"] == "Put strong emphasis on the word 'big'."

    def test_emphasis_clause_names_multiple_spans_in_original_form(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("*Not* the *whole* story", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "NOT the WHOLE story"
        assert call["instruct"] == "Put strong emphasis on the words 'Not' and 'whole'."

    def test_short_lowercase_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 2026-07-15 audition: CAPS turned 'it' into the acronym "eye-tee", and
        # clause-alone landed the target word 0-for-5 across both audition rounds
        # (MANIFEST.md / ADR 0014) — so a short lowercase span is left alone.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's *it* — the entire fix was one line.", "Ryan", tmp_path / "x.wav"
        )
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "That's it — the entire fix was one line."
        assert call["instruct"] is None

    def test_short_lowercase_span_leaves_delivery_note_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "That's *it*.", "Ryan", tmp_path / "x.wav", delivery="excited, racing ahead"
        )
        assert _FakeModel.generate_calls[0]["instruct"] == "excited, racing ahead"

    def test_all_caps_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Round-2 audition: the clause without a CAPS change stressed the wrong
        # word ("not" instead of RAG) — clause-alone went 0-for-5 across rounds.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line(
            "Plain *RAG* retrieves neighbors from a vector store.", "Ryan", tmp_path / "x.wav"
        )
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "Plain RAG retrieves neighbors from a vector store."
        assert call["instruct"] is None

    def test_two_char_all_caps_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Round-2 audition: the clause-alone treatment of *AI* was inaudible.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("The *AI* did this.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "The AI did this."
        assert call["instruct"] is None

    def test_numeric_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Uppercasing cannot change a numeral, so it gets no clause either.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("All *100* queries hit.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "All 100 queries hit."
        assert call["instruct"] is None

    def test_punctuated_short_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # '*it.*' measures 3 chars, but the audition rule applies to the span's
        # word content — the 2-char 'it' — so trailing punctuation cannot smuggle
        # the acronym misread ("eye-tee") past the short-span guard.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("That's *it.* One line.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "That's it. One line."
        assert call["instruct"] is None

    def test_punctuated_all_caps_span_gets_no_treatment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Word content 'RAG' is all-caps: CAPS cannot change it, so no treatment.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("Try plain *RAG.* Then compare.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "Try plain RAG. Then compare."
        assert call["instruct"] is None

    def test_punctuated_word_keeps_caps_and_clause_names_the_full_span(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Word content 'Not' (3 alnum chars, CAPS changes it) is treated; the
        # clause quotes the span exactly as written, punctuation included.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("*Not!* every query needs it.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "NOT! every query needs it."
        assert call["instruct"] == "Put strong emphasis on the word 'Not!'."

    def test_multi_word_span_is_treated_on_its_joined_word_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 'two words' has word content 'twowords' — multi-word spans stay treated.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("the *two words* stand", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "the TWO WORDS stand"
        assert call["instruct"] == "Put strong emphasis on the word 'two words'."

    def test_mixed_line_treats_only_spans_caps_can_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("That's *it*, the *reranker* wins.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "That's it, the RERANKER wins."
        assert call["instruct"] == "Put strong emphasis on the word 'reranker'."

    def test_capitalized_span_keeps_caps_and_clause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Audition winner on mixed case: *Not* renders as NOT and is named as written.
        _install_fakes(monkeypatch, cuda_available=True)
        engine = qwen3.Qwen3Engine(AppConfig())
        engine.synthesize_line("*Not* every query needs retrieval.", "Ryan", tmp_path / "x.wav")
        call = _FakeModel.generate_calls[0]
        assert call["text"] == "NOT every query needs retrieval."
        assert call["instruct"] == "Put strong emphasis on the word 'Not'."

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
