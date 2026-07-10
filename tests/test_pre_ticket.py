"""Tests for scripts/pre_ticket.py."""

from collections.abc import Sequence

import pytest

import gatelib
import pre_ticket


class TestMain:
    def test_green_baseline_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        commands: list[Sequence[str]] = []

        def runner(command: Sequence[str]) -> int:
            commands.append(command)
            return 0

        assert pre_ticket.main(["PHASE-2"], runner=runner) == 0
        assert len(commands) == len(gatelib.default_gates())
        assert "pre-ticket PHASE-2: OK" in capsys.readouterr().out

    def test_dirty_baseline_returns_one(self) -> None:
        assert pre_ticket.main(["PHASE-2"], runner=lambda _command: 1) == 1

    def test_ticket_id_is_required(self) -> None:
        with pytest.raises(SystemExit):
            pre_ticket.main([])
