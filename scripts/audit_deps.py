# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Dependency audit gate: pip-audit over the locked PyPI dependency set.

Audits `uv export` output (project + dev dependencies, no extras) instead of the
local environment: the qwen3 extra installs AMD TheRock packages that do not
exist on PyPI and would otherwise abort the audit.
"""

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

type CommandRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def run_command(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    # S603: fixed argv from shutil.which/sys.executable — no untrusted input.
    return subprocess.run(list(command), capture_output=True, text=True, check=False)  # noqa: S603


def main(runner: CommandRunner = run_command) -> int:
    uv = shutil.which("uv")
    if uv is None:
        print("uv not found on PATH")
        return 1
    export = runner([uv, "export", "--format", "requirements-txt", "--no-emit-project"])
    if export.returncode != 0:
        print(export.stderr.strip())
        return 1
    with tempfile.TemporaryDirectory() as scratch:
        requirements = Path(scratch) / "requirements.txt"
        requirements.write_text(export.stdout, encoding="utf-8")
        audit = runner([sys.executable, "-m", "pip_audit", "-r", str(requirements), "--no-deps"])
    print(audit.stdout.strip() or audit.stderr.strip())
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
