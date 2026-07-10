"""Shared quality-gate runner for the gate scripts (standards v3)."""

import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

type CommandRunner = Callable[[Sequence[str]], int]


@dataclass(frozen=True)
class Gate:
    """One named quality gate and the argv that runs it."""

    name: str
    command: tuple[str, ...]


def default_gates() -> tuple[Gate, ...]:
    checked = ("src", "tests", "scripts")
    return (
        Gate("ruff check", (sys.executable, "-m", "ruff", "check", *checked)),
        Gate("ruff format", (sys.executable, "-m", "ruff", "format", "--check", *checked)),
        Gate("mypy", (sys.executable, "-m", "mypy")),
        Gate("pyright", (sys.executable, "-m", "pyright")),
        Gate("pytest", (sys.executable, "-m", "pytest", "--cov")),
    )


def lock_check_gate() -> Gate | None:
    """Lockfile integrity gate; skipped (None) when uv is not installed."""
    uv = shutil.which("uv")
    if uv is None:
        return None
    return Gate("uv lock --check", (uv, "lock", "--check"))


def run_command(command: Sequence[str]) -> int:
    # S603: fixed argv built from sys.executable/shutil.which — no untrusted input.
    return subprocess.run(list(command), cwd=REPO_ROOT, check=False).returncode  # noqa: S603


def run_gates(gates: Sequence[Gate], runner: CommandRunner = run_command) -> list[tuple[Gate, int]]:
    results: list[tuple[Gate, int]] = []
    for gate in gates:
        print(f"==> {gate.name}", flush=True)
        results.append((gate, runner(gate.command)))
    return results


def report(results: Sequence[tuple[Gate, int]], label: str) -> int:
    """Print a pass/fail summary; return a shell exit code."""
    failed = [gate.name for gate, code in results if code != 0]
    if failed:
        print(f"{label}: FAIL ({', '.join(failed)})")
        return 1
    print(f"{label}: OK ({len(results)} gates)")
    return 0
