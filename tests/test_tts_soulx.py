"""Tests for podcast.tts.soulx (SoulX runtime replaced by typed doubles)."""

import sys
import types
import wave
from pathlib import Path
from typing import ClassVar

import pytest

from podcast.config import AppConfig, PathsSettings, TTSSettings
from podcast.errors import TTSError
from podcast.tts import soulx
from podcast.tts.base import DialogueLine


class _FakeWavTensor:
    def __init__(self, samples: list[float]) -> None:
        self._samples = samples

    def squeeze(self, _axis: int) -> "_FakeWavTensor":
        return self

    def cpu(self) -> "_FakeWavTensor":
        return self

    def numpy(self) -> list[float]:
        return self._samples


class _FakeModel:
    forward_calls: "ClassVar[list[dict[str, object]]]" = []
    turns_out = 0

    def forward_longform(self, **data: object) -> dict[str, list[_FakeWavTensor]]:
        _FakeModel.forward_calls.append(data)
        return {"generated_wavs": [_FakeWavTensor([0.0, 0.5]) for _ in range(self.turns_out)]}


_PROCESS_CALLS: list[tuple[object, ...]] = []


def _fake_process(*args: object) -> dict[str, object]:
    _PROCESS_CALLS.append(args)
    return {"payload": "x"}


def _install_fakes(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> None:
    _FakeModel.forward_calls = []
    _FakeModel.turns_out = 0
    _PROCESS_CALLS.clear()

    def _snapshot(_model_id: str) -> str:
        return "/weights"

    def _initiate(_seed: int, _path: str, _engine: str, _fp16: bool) -> tuple[_FakeModel, str]:
        return _FakeModel(), "dataset-sentinel"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _snapshot  # type: ignore[attr-defined]
    cli = types.ModuleType("cli")
    cli_podcast = types.ModuleType("cli.podcast")
    cli_podcast.initiate_model = _initiate  # type: ignore[attr-defined]
    sp = types.ModuleType("soulxpodcast")
    sp_utils = types.ModuleType("soulxpodcast.utils")
    sp_infer = types.ModuleType("soulxpodcast.utils.infer_utils")
    sp_infer.process_single_input = _fake_process  # type: ignore[attr-defined]
    for name, module in (
        ("huggingface_hub", hub),
        ("cli", cli),
        ("cli.podcast", cli_podcast),
        ("soulxpodcast", sp),
        ("soulxpodcast.utils", sp_utils),
        ("soulxpodcast.utils.infer_utils", sp_infer),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def _repo(_models_dir: Path) -> Path:
        return models_dir / "soulx" / "SoulX-Podcast"

    monkeypatch.setattr(soulx, "shim_torchaudio", lambda: None)
    monkeypatch.setattr(soulx, "ensure_repo", _repo)


def _config(tmp_path: Path) -> AppConfig:
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("alex", "maya"):
        (refs_dir / f"{name}.wav").write_bytes(b"RIFF")
        (refs_dir / f"{name}.txt").write_text("Reference transcript.", encoding="utf-8")
    return AppConfig(
        paths=PathsSettings(models_dir=tmp_path / "models"),
        tts=TTSSettings(
            engine="soulx",
            soulx_refs={
                "alex": str(refs_dir / "alex.wav"),
                "maya": str(refs_dir / "maya.wav"),
            },
        ),
    )


def _lines() -> list[DialogueLine]:
    return [
        DialogueLine(speaker="Alex", text="Hello there.", delivery=""),
        DialogueLine(speaker="Maya", text="Get this.", delivery="excited, laughing"),
    ]


VOICES = {"Alex": "alex", "Maya": "maya"}


class TestTaggedText:
    def test_plain_delivery_is_untouched(self) -> None:
        assert soulx.tagged_text(DialogueLine(speaker="A", text="Hi.")) == "Hi."

    def test_keywords_map_to_tags(self) -> None:
        line = DialogueLine(speaker="A", text="Hi.", delivery="laughing, then a sigh")
        assert soulx.tagged_text(line) == "<|laughter|><|sigh|>Hi."


class TestEnsureRepo:
    def test_clones_once_then_checks_out_pin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_git(*args: str) -> None:
            calls.append(args)
            if args[0] == "clone":
                (Path(args[-1]) / "soulxpodcast").mkdir(parents=True)

        monkeypatch.setattr(soulx, "_git", fake_git)
        repo = soulx.ensure_repo(tmp_path)
        assert repo == tmp_path / "soulx" / "SoulX-Podcast"
        assert calls[0][0] == "clone"
        assert calls[1][2:] == ("checkout", "--quiet", soulx.REPO_COMMIT)
        soulx.ensure_repo(tmp_path)  # second call: checkout only
        assert [call[0] for call in calls] == ["clone", "-C", "-C"]

    def test_git_failure_raises_tts_error(self, tmp_path: Path) -> None:
        with pytest.raises(TTSError, match="cannot fetch SoulX source"):
            soulx._git("clone", "file:///nonexistent-src", str(tmp_path / "dst"))  # pyright: ignore[reportPrivateUsage]

    def test_git_success_returns_quietly(self) -> None:
        soulx._git("--version")  # pyright: ignore[reportPrivateUsage]


class TestShimTorchaudio:
    def test_load_and_save_route_through_soundfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class _Array:
            T = "transposed"

        writes: list[tuple[str, object, int]] = []
        soundfile = types.ModuleType("soundfile")

        def _read(_path: str, **_kw: object) -> tuple[_Array, int]:
            return _Array(), 24000

        def _write(path: str, data: object, sr: int) -> None:
            writes.append((path, data, sr))

        soundfile.read = _read  # type: ignore[attr-defined]
        soundfile.write = _write  # type: ignore[attr-defined]
        torch = types.ModuleType("torch")
        torch.from_numpy = lambda value: value  # type: ignore[attr-defined]
        torchaudio = types.ModuleType("torchaudio")
        monkeypatch.setitem(sys.modules, "soundfile", soundfile)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)

        soulx.shim_torchaudio()
        tensor, sr = torchaudio.load(tmp_path / "x.wav")
        assert (tensor, sr) == ("transposed", 24000)

        class _Tensor:
            def detach(self) -> "_Tensor":
                return self

            def cpu(self) -> "_Tensor":
                return self

            def numpy(self) -> _Array:
                return _Array()

        torchaudio.save(tmp_path / "y.wav", _Tensor(), 24000)
        assert writes[0][1] == "transposed"
        assert writes[0][2] == 24000


class TestSoulXEngine:
    def test_info_flags(self, tmp_path: Path) -> None:
        engine = soulx.SoulXEngine(_config(tmp_path))
        info = engine.info()
        assert info.dialogue_native is True
        assert info.supports_delivery is True
        assert info.sample_rate == soulx.SAMPLE_RATE

    def test_missing_extra_raises_install_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def broken_shim() -> None:
            raise ImportError("no soundfile")

        monkeypatch.setattr(soulx, "shim_torchaudio", broken_shim)
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="uv sync --extra soulx"):
            engine.synthesize_dialogue(_lines(), VOICES, [tmp_path / "a.wav", tmp_path / "b.wav"])

    def test_dialogue_renders_one_wav_per_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        _FakeModel.turns_out = 3
        engine = soulx.SoulXEngine(_config(tmp_path))
        outs = [tmp_path / "0.wav", tmp_path / "1.wav", tmp_path / "2.wav"]
        lines = [*_lines(), DialogueLine(speaker="Alex", text="Again me.")]
        engine.synthesize_dialogue(lines, VOICES, outs)
        texts = _PROCESS_CALLS[0][1]
        assert texts == ["[S1]Hello there.", "[S2]<|laughter|>Get this.", "[S1]Again me."]
        for out in outs:
            with wave.open(str(out), "rb") as handle:
                assert handle.getnframes() == 2

    def test_second_engine_reuses_sys_path_entry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        _FakeModel.turns_out = 1
        line = [DialogueLine(speaker="A", text="hi")]
        for index in range(2):
            engine = soulx.SoulXEngine(_config(tmp_path))
            engine.synthesize_dialogue(line, {"A": "alex"}, [tmp_path / f"p{index}.wav"])

    def test_broken_checkout_raises_import_error_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        monkeypatch.delitem(sys.modules, "cli.podcast")
        monkeypatch.delitem(sys.modules, "cli")
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="failed to import"):
            engine.synthesize_dialogue(
                [DialogueLine(speaker="A", text="hi")], {"A": "alex"}, [tmp_path / "0.wav"]
            )

    def test_tts_error_from_pipeline_passes_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")

        def reject(*_args: object) -> dict[str, object]:
            raise TTSError("prompt wav unreadable")

        monkeypatch.setitem(
            sys.modules["soulxpodcast.utils.infer_utils"].__dict__, "process_single_input", reject
        )
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="prompt wav unreadable"):
            engine.synthesize_dialogue(
                [DialogueLine(speaker="A", text="hi")], {"A": "alex"}, [tmp_path / "0.wav"]
            )

    def test_model_loads_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        _FakeModel.turns_out = 2
        engine = soulx.SoulXEngine(_config(tmp_path))
        engine.synthesize_dialogue(_lines(), VOICES, [tmp_path / "0.wav", tmp_path / "1.wav"])
        engine.synthesize_dialogue(_lines(), VOICES, [tmp_path / "2.wav", tmp_path / "3.wav"])
        assert len(_PROCESS_CALLS) == 2

    def test_turn_count_mismatch_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        _FakeModel.turns_out = 1
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="returned 1 turns for 2 lines"):
            engine.synthesize_dialogue(_lines(), VOICES, [tmp_path / "0.wav", tmp_path / "1.wav"])

    def test_runtime_failure_maps_to_tts_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")

        def explode(*_args: object) -> dict[str, object]:
            raise RuntimeError("hip error")

        monkeypatch.setitem(
            sys.modules["soulxpodcast.utils.infer_utils"].__dict__, "process_single_input", explode
        )
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="soulx failed"):
            engine.synthesize_dialogue(_lines(), VOICES, [tmp_path / "0.wav", tmp_path / "1.wav"])

    def test_too_many_speakers_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        lines = [
            DialogueLine(speaker=f"H{index}", text="hi") for index in range(soulx.MAX_SPEAKERS + 1)
        ]
        voices = {f"H{index}": "alex" for index in range(soulx.MAX_SPEAKERS + 1)}
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="at most 4 speakers"):
            engine.synthesize_dialogue(lines, voices, [tmp_path / f"{i}.wav" for i in range(5)])

    def test_unknown_reference_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        engine = soulx.SoulXEngine(_config(tmp_path))
        with pytest.raises(TTSError, match="no SoulX reference for voice 'ghost'"):
            engine.synthesize_dialogue(
                [DialogueLine(speaker="A", text="hi")], {"A": "ghost"}, [tmp_path / "0.wav"]
            )

    def test_missing_reference_file_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        config = _config(tmp_path)
        config.tts.soulx_refs["alex"] = str(tmp_path / "gone.wav")
        engine = soulx.SoulXEngine(config)
        with pytest.raises(TTSError, match="not found"):
            engine.synthesize_dialogue(
                [DialogueLine(speaker="A", text="hi")], {"A": "alex"}, [tmp_path / "0.wav"]
            )

    def test_synthesize_line_delegates_to_dialogue(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _install_fakes(monkeypatch, tmp_path / "models")
        _FakeModel.turns_out = 1
        engine = soulx.SoulXEngine(_config(tmp_path))
        out = tmp_path / "line.wav"
        engine.synthesize_line("Just one line.", "maya", out, delivery="a sigh")
        assert _PROCESS_CALLS[0][1] == ["[S1]<|sigh|>Just one line."]
        assert out.is_file()
