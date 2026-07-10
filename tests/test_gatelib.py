"""Tests for scripts/gatelib.py."""

import sys
from collections.abc import Sequence

import pytest

import gatelib


def _which_uv(_name: str) -> str:
    return "/usr/bin/uv"


def _which_missing(_name: str) -> None:
    return None


class TestDefaultGates:
    def test_covers_all_five_gates(self) -> None:
        names = [gate.name for gate in gatelib.default_gates()]
        assert names == ["ruff check", "ruff format", "mypy", "pyright", "pytest"]

    def test_all_commands_use_current_interpreter(self) -> None:
        for gate in gatelib.default_gates():
            assert gate.command[0] == sys.executable


class TestLockCheckGate:
    def test_none_when_uv_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("gatelib.shutil.which", _which_missing)
        assert gatelib.lock_check_gate() is None

    def test_gate_when_uv_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("gatelib.shutil.which", _which_uv)
        gate = gatelib.lock_check_gate()
        assert gate is not None
        assert gate.command == ("/usr/bin/uv", "lock", "--check")


class TestRunCommand:
    def test_returns_child_exit_code(self) -> None:
        code = gatelib.run_command([sys.executable, "-c", "raise SystemExit(3)"])
        assert code == 3

    def test_zero_for_success(self) -> None:
        assert gatelib.run_command([sys.executable, "-c", "pass"]) == 0


class TestRunGates:
    def test_runs_every_gate_and_collects_codes(self, capsys: pytest.CaptureFixture[str]) -> None:
        seen: list[Sequence[str]] = []

        def fake_runner(command: Sequence[str]) -> int:
            seen.append(command)
            return len(seen) - 1

        gates = (gatelib.Gate("a", ("x",)), gatelib.Gate("b", ("y",)))
        results = gatelib.run_gates(gates, fake_runner)
        assert [code for _, code in results] == [0, 1]
        assert seen == [("x",), ("y",)]
        output = capsys.readouterr().out
        assert "==> a" in output
        assert "==> b" in output


class TestReport:
    def test_all_green_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [(gatelib.Gate("a", ("x",)), 0), (gatelib.Gate("b", ("y",)), 0)]
        assert gatelib.report(results, "phase-1") == 0
        assert "phase-1: OK (2 gates)" in capsys.readouterr().out

    def test_failures_are_named(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [(gatelib.Gate("a", ("x",)), 0), (gatelib.Gate("b", ("y",)), 2)]
        assert gatelib.report(results, "phase-1") == 1
        assert "FAIL (b)" in capsys.readouterr().out
