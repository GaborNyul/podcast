"""Tests for scripts/post_ticket.py."""

from collections.abc import Sequence

import pytest

import gatelib
import post_ticket


def _which_uv(_name: str) -> str:
    return "/usr/bin/uv"


def _which_missing(_name: str) -> None:
    return None


class TestMain:
    def test_includes_lock_check_when_uv_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("gatelib.shutil.which", _which_uv)
        commands: list[Sequence[str]] = []

        def runner(command: Sequence[str]) -> int:
            commands.append(command)
            return 0

        assert post_ticket.main(["PHASE-2"], runner=runner) == 0
        assert len(commands) == len(gatelib.default_gates()) + 1
        assert commands[-1] == ("/usr/bin/uv", "lock", "--check")

    def test_skips_lock_check_without_uv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("gatelib.shutil.which", _which_missing)
        commands: list[Sequence[str]] = []

        def runner(command: Sequence[str]) -> int:
            commands.append(command)
            return 0

        assert post_ticket.main(["PHASE-2"], runner=runner) == 0
        assert len(commands) == len(gatelib.default_gates())

    def test_gate_failure_returns_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("gatelib.shutil.which", _which_missing)
        assert post_ticket.main(["PHASE-2"], runner=lambda _command: 2) == 1
