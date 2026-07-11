"""Environment smoke tests: can this machine actually run the tool?"""

import os
import shutil
import subprocess
from dataclasses import dataclass

from podcast.config import AppConfig
from podcast.errors import PodcastError


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one doctor check; `hint` tells the user how to fix a failure."""

    name: str
    ok: bool
    detail: str
    hint: str = ""


def check_ffmpeg() -> CheckResult:
    path = shutil.which("ffmpeg")
    if path is None:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            detail="not found on PATH",
            hint="install ffmpeg (e.g. sudo apt install ffmpeg)",
        )
    # S603: fixed argv, absolute path resolved via shutil.which — no untrusted input.
    proc = subprocess.run(  # noqa: S603
        [path, "-version"], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            detail=f"{path} exists but failed to run",
            hint="reinstall ffmpeg",
        )
    version = proc.stdout.splitlines()[0] if proc.stdout else "unknown version"
    return CheckResult(name="ffmpeg", ok=True, detail=version)


def check_episodes_dir(config: AppConfig) -> CheckResult:
    directory = config.paths.episodes_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".doctor-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult(
            name="episodes dir",
            ok=False,
            detail=f"{directory} is not writable: {exc}",
            hint="set [paths].episodes_dir in podcast.toml to a writable location",
        )
    return CheckResult(name="episodes dir", ok=True, detail=str(directory))


def check_models_dir(config: AppConfig) -> CheckResult:
    directory = config.paths.resolved_models_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return CheckResult(
            name="models dir",
            ok=False,
            detail=f"{directory} cannot be created: {exc}",
            hint="set [paths].models_dir in podcast.toml to a writable location",
        )
    return CheckResult(name="models dir", ok=True, detail=str(directory))


def check_kokoro(config: AppConfig) -> CheckResult:
    try:
        import kokoro_onnx  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as exc:
        return CheckResult(
            name="kokoro engine",
            ok=False,
            detail=f"kokoro-onnx not importable: {exc}",
            hint="reinstall the project (`uv sync`)",
        )
    models_dir = config.paths.resolved_models_dir()
    have_files = (models_dir / "kokoro-v1.0.onnx").is_file()
    detail = "ready" if have_files else "installed; model files download on first use (~310 MB)"
    return CheckResult(name="kokoro engine", ok=True, detail=detail)


def check_qwen3(config: AppConfig) -> CheckResult:
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return CheckResult(
            name="qwen3 engine",
            ok=False,
            detail="HSA_OVERRIDE_GFX_VERSION is set",
            hint="unset it — gfx1151 must NOT masquerade as gfx1100 (see README)",
        )
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError:
        return CheckResult(
            name="qwen3 engine",
            ok=False,
            detail="torch not installed",
            hint="run `uv sync --extra qwen3` to pull TheRock ROCm wheels",
        )
    cuda_ok = bool(
        torch.cuda.is_available()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    if not cuda_ok:
        return CheckResult(
            name="qwen3 engine",
            ok=False,
            detail="torch installed but no GPU visible",
            hint="check TheRock wheel install and amdgpu driver (see README)",
        )
    device_name = str(
        torch.cuda.get_device_name(0)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    device = config.tts.device or "cuda"
    return CheckResult(name="qwen3 engine", ok=True, detail=f"{device_name} via {device}")


def check_soulx(config: AppConfig) -> CheckResult:
    try:
        import s3tokenizer  # noqa: F401  # pyright: ignore[reportMissingImports, reportMissingTypeStubs, reportUnusedImport]
    except ImportError:
        return CheckResult(
            name="soulx engine",
            ok=False,
            detail="s3tokenizer not installed",
            hint="run `uv sync --extra soulx`",
        )
    from podcast.tts.soulx import SoulXEngine
    from podcast.tts.voices import voices_for

    engine = SoulXEngine(config)
    try:
        for voice in voices_for("soulx"):
            engine.cache_token(voice.id)
    except PodcastError as exc:
        return CheckResult(
            name="soulx engine",
            ok=False,
            detail=str(exc),
            hint="point [tts.soulx_refs] at a reference WAV + .txt transcript per voice",
        )
    return CheckResult(name="soulx engine", ok=True, detail="clone references and deps present")


def check_engine(config: AppConfig) -> CheckResult:
    if config.tts.engine == "kokoro":
        return check_kokoro(config)
    if config.tts.engine == "qwen3":
        return check_qwen3(config)
    if config.tts.engine == "soulx":
        return check_soulx(config)
    return CheckResult(
        name="tts engine",
        ok=False,
        detail=f"unknown engine {config.tts.engine!r}",
        hint="set [tts].engine to kokoro, qwen3, or soulx",
    )


def run_checks(config: AppConfig) -> list[CheckResult]:
    """All doctor checks for the configured setup."""
    return [
        check_ffmpeg(),
        check_episodes_dir(config),
        check_models_dir(config),
        check_engine(config),
    ]
