"""Tests for podcast.audio.assemble (ffmpeg replaced by a recording double)."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from podcast.audio import assemble
from podcast.errors import AudioError


def _which_found(_name: str) -> str:
    return "/usr/bin/ffmpeg"


def _which_missing(_name: str) -> None:
    return None


class _FakeFfmpeg:
    """Records every argv and creates the output file each command names last."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.fail_with: str | None = None

    def __call__(
        self, command: Sequence[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        self.commands.append(argv)
        if self.fail_with is not None:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=self.fail_with)
        Path(argv[-1]).write_bytes(b"fake-audio")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


@pytest.fixture
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> _FakeFfmpeg:
    fake = _FakeFfmpeg()
    monkeypatch.setattr("podcast.audio.assemble.shutil.which", _which_found)
    monkeypatch.setattr("podcast.audio.assemble.subprocess.run", fake)
    return fake


def _segments(tmp_path: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = tmp_path / f"seg{index}.wav"
        path.write_bytes(b"RIFF")
        paths.append(path)
    return paths


def _assemble(tmp_path: Path, segments: Sequence[Path], **overrides: object) -> Path:
    out = tmp_path / "episode.mp3"
    kwargs: dict[str, object] = {
        "work_dir": tmp_path / "work",
        "sample_rate": 24000,
        "pause_min_ms": 200,
        "pause_max_ms": 1000,
        "bitrate": "192k",
        "seed": 7,
    }
    kwargs.update(overrides)
    assemble.assemble_episode(segments, out, **kwargs)  # type: ignore[arg-type]
    return out


class TestFindFfmpeg:
    def test_missing_binary_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("podcast.audio.assemble.shutil.which", _which_missing)
        with pytest.raises(AudioError, match="ffmpeg not found"):
            assemble.find_ffmpeg()


class TestAssembleEpisode:
    def test_no_segments_raises(self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg) -> None:
        del fake_ffmpeg
        with pytest.raises(AudioError, match="no audio segments"):
            _assemble(tmp_path, [])

    @pytest.mark.usefixtures("fake_ffmpeg")
    def test_concat_list_interleaves_silence(self, tmp_path: Path) -> None:
        segments = _segments(tmp_path, 3)
        _assemble(tmp_path, segments)
        concat = (tmp_path / "work" / "concat.txt").read_text(encoding="utf-8")
        lines = concat.strip().splitlines()
        assert len(lines) == 5  # 3 segments + 2 gaps
        assert "seg0.wav" in lines[0]
        assert "silence-" in lines[1]
        assert "seg1.wav" in lines[2]

    @pytest.mark.usefixtures("fake_ffmpeg")
    def test_single_segment_has_no_silence(self, tmp_path: Path) -> None:
        _assemble(tmp_path, _segments(tmp_path, 1))
        concat = (tmp_path / "work" / "concat.txt").read_text(encoding="utf-8")
        assert "silence" not in concat

    @pytest.mark.usefixtures("fake_ffmpeg")
    def test_seed_makes_gaps_deterministic(self, tmp_path: Path) -> None:
        segments = _segments(tmp_path, 4)
        _assemble(tmp_path, segments, work_dir=tmp_path / "w1", seed=42)
        _assemble(tmp_path, segments, work_dir=tmp_path / "w2", seed=42)
        first = (tmp_path / "w1" / "concat.txt").read_text(encoding="utf-8")
        second = (tmp_path / "w2" / "concat.txt").read_text(encoding="utf-8")
        assert [line.split("/")[-1] for line in first.splitlines()] == [
            line.split("/")[-1] for line in second.splitlines()
        ]

    def test_silence_durations_within_configured_range(
        self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg
    ) -> None:
        del fake_ffmpeg
        _assemble(tmp_path, _segments(tmp_path, 10), pause_min_ms=200, pause_max_ms=400)
        concat = (tmp_path / "work" / "concat.txt").read_text(encoding="utf-8")
        durations = [
            int(line.split("silence-")[1].split("ms")[0])
            for line in concat.splitlines()
            if "silence-" in line
        ]
        assert durations
        assert all(150 <= duration <= 450 for duration in durations)

    def test_loudnorm_and_bitrate_in_final_command(
        self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg
    ) -> None:
        _assemble(tmp_path, _segments(tmp_path, 2), bitrate="128k")
        final = fake_ffmpeg.commands[-1]
        assert assemble.LOUDNORM_FILTER in final
        assert "128k" in final
        assert final[-1].endswith("episode.mp3")

    def test_concat_uses_demuxer_and_resamples(
        self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg
    ) -> None:
        _assemble(tmp_path, _segments(tmp_path, 2))
        concat_command = next(command for command in fake_ffmpeg.commands if "concat" in command)
        assert "-ar" in concat_command
        assert "24000" in concat_command

    def test_equal_pause_bounds_skip_randomness(
        self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg
    ) -> None:
        del fake_ffmpeg
        _assemble(tmp_path, _segments(tmp_path, 3), pause_min_ms=300, pause_max_ms=300)
        concat = (tmp_path / "work" / "concat.txt").read_text(encoding="utf-8")
        assert concat.count("silence-300ms.wav") == 2

    def test_ffmpeg_failure_surfaces_stderr(self, tmp_path: Path, fake_ffmpeg: _FakeFfmpeg) -> None:
        fake_ffmpeg.fail_with = "Invalid data found when processing input"
        with pytest.raises(AudioError, match="Invalid data found"):
            _assemble(tmp_path, _segments(tmp_path, 2))
