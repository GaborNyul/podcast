"""Environment smoke tests: can this machine actually run the tool?"""

import shutil
import subprocess
from dataclasses import dataclass

from podcast.config import AppConfig


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


def run_checks(config: AppConfig) -> list[CheckResult]:
    """All doctor checks; engine-specific checks register here in later phases."""
    return [
        check_ffmpeg(),
        check_episodes_dir(config),
        check_models_dir(config),
    ]
