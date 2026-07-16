# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for scripts/adversarial_review.py."""

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import adversarial_review as ar
import gatelib


def _finding(identifier: str, severity: ar.Severity) -> ar.ReviewFinding:
    return ar.ReviewFinding(
        identifier=identifier, description=f"fault {identifier}", severity=severity
    )


def _config(max_iterations: int = 5) -> ar.LoopConfig:
    return ar.LoopConfig(max_iterations=max_iterations)


def _which_claude(_name: str) -> str:
    return "/usr/bin/claude"


def _which_missing(_name: str) -> None:
    return None


def _no_fix(_findings: Sequence[ar.ReviewFinding]) -> None:
    return None


def _gates_green() -> bool:
    return True


def _gates_red() -> bool:
    return False


def _fake_run_gates(
    code: int,
) -> "Callable[[Sequence[gatelib.Gate]], list[tuple[gatelib.Gate, int]]]":
    def runner(gates: Sequence[gatelib.Gate]) -> list[tuple[gatelib.Gate, int]]:
        return [(gate, code) for gate in gates]

    return runner


class _ScriptedReviewer:
    """Returns a pre-planned findings list per round; empty after the plan runs out."""

    def __init__(self, rounds: Sequence[list[ar.ReviewFinding]]) -> None:
        self.rounds = list(rounds)
        self.calls: list[int] = []

    def __call__(self, round_no: int, _known: Sequence[ar.ReviewFinding]) -> list[ar.ReviewFinding]:
        self.calls.append(round_no)
        if round_no <= len(self.rounds):
            return self.rounds[round_no - 1]
        return []


class TestMustFix:
    @pytest.mark.parametrize(
        ("severity", "threshold", "expected"),
        [
            ("critical", "high", True),
            ("high", "high", True),
            ("medium", "high", False),
            ("low", "high", False),
            ("high", "critical", False),
            ("critical", "critical", True),
            ("low", "low", True),
        ],
    )
    def test_threshold_ordering(
        self, severity: ar.Severity, threshold: ar.Severity, expected: bool
    ) -> None:
        assert ar.must_fix(_finding("x", severity), threshold) is expected


class TestRunLoop:
    def test_clean_first_round_converges(self) -> None:
        reviewer = _ScriptedReviewer([[]])
        outcome = ar.run_loop(_config(), reviewer, _no_fix, _gates_green)
        assert outcome.stop_reason == "converged"
        assert outcome.rounds_run == 1
        assert outcome.passed
        assert outcome.findings == []

    def test_fixes_critical_and_records_low_as_residual(self) -> None:
        reviewer = _ScriptedReviewer([[_finding("crash", "critical"), _finding("style", "low")]])
        fixed: list[list[str]] = []

        def fixer(findings: Sequence[ar.ReviewFinding]) -> None:
            fixed.append([item.identifier for item in findings])

        outcome = ar.run_loop(_config(), reviewer, fixer, _gates_green)
        assert outcome.stop_reason == "converged"
        assert outcome.rounds_run == 2
        assert fixed == [["crash"]]
        statuses = {item.finding.identifier: item.status for item in outcome.findings}
        assert statuses == {"crash": "fixed", "style": "residual"}
        assert outcome.passed

    def test_gate_breaking_fix_counts_as_no_progress(self) -> None:
        reviewer = _ScriptedReviewer([[_finding("crash", "high")]])
        gate_results = iter([False, True])

        def gates() -> bool:
            return next(gate_results)

        fix_calls: list[int] = []

        def fixer(findings: Sequence[ar.ReviewFinding]) -> None:
            fix_calls.append(len(findings))

        outcome = ar.run_loop(_config(), reviewer, fixer, gates)
        assert outcome.stop_reason == "converged"
        assert outcome.rounds_run == 3
        assert fix_calls == [1, 1]
        assert outcome.passed

    def test_cap_reached_escalates_open_findings(self) -> None:
        rounds = [[_finding(f"bug-{index}", "critical")] for index in range(1, 6)]
        reviewer = _ScriptedReviewer(rounds)
        outcome = ar.run_loop(_config(max_iterations=3), reviewer, _no_fix, _gates_red)
        assert outcome.stop_reason == "cap-reached"
        assert outcome.rounds_run == 3
        assert not outcome.passed
        assert all(item.status == "escalated" for item in outcome.findings)
        assert all("human sign-off" in item.reason_unfixed for item in outcome.findings)

    def test_resurfaced_fixed_finding_halts_as_oscillation(self) -> None:
        finding = _finding("flip-flop", "high")
        reviewer = _ScriptedReviewer([[finding], [finding]])
        outcome = ar.run_loop(_config(), reviewer, _no_fix, _gates_green)
        assert outcome.stop_reason == "oscillation"
        assert outcome.rounds_run == 2
        assert not outcome.passed
        tracked = outcome.findings[0]
        assert tracked.status == "escalated"
        assert "resurfaced" in tracked.reason_unfixed

    def test_medium_findings_never_block_convergence(self) -> None:
        reviewer = _ScriptedReviewer([[_finding("meh", "medium")]])
        fix_calls: list[int] = []

        def fixer(findings: Sequence[ar.ReviewFinding]) -> None:
            fix_calls.append(len(findings))

        outcome = ar.run_loop(_config(), reviewer, fixer, _gates_green)
        assert outcome.stop_reason == "converged"
        assert outcome.rounds_run == 1
        assert fix_calls == []
        assert outcome.findings[0].status == "residual"
        assert outcome.passed

    def test_re_reported_open_finding_is_not_duplicated(self) -> None:
        finding = _finding("sticky", "high")
        reviewer = _ScriptedReviewer([[finding], [finding], []])
        gate_results = iter([False, True])

        def gates() -> bool:
            return next(gate_results)

        outcome = ar.run_loop(_config(), reviewer, _no_fix, gates)
        assert outcome.stop_reason == "converged"
        assert len(outcome.findings) == 1
        assert outcome.findings[0].status == "fixed"


class TestParseFindings:
    def test_parses_bare_json_array(self) -> None:
        raw = json.dumps([{"identifier": "a", "description": "d", "severity": "high"}])
        findings = ar.parse_findings(raw)
        assert findings[0].identifier == "a"

    def test_parses_array_surrounded_by_prose(self) -> None:
        raw = 'Here you go:\n[{"identifier": "a", "description": "d", "severity": "low"}]\nDone.'
        assert len(ar.parse_findings(raw)) == 1

    def test_empty_array_is_valid(self) -> None:
        assert ar.parse_findings("[]") == []

    def test_missing_array_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON array"):
            ar.parse_findings("all good, nothing to report")

    def test_invalid_severity_raises(self) -> None:
        raw = '[{"identifier": "a", "description": "d", "severity": "catastrophic"}]'
        with pytest.raises(ValueError, match="failed validation"):
            ar.parse_findings(raw)


class TestWriteReport:
    def test_writes_full_audit_trail(self, tmp_path: Path) -> None:
        outcome = ar.LoopOutcome(
            stop_reason="converged",
            rounds_run=2,
            findings=[
                ar.TrackedFinding(_finding("crash", "critical"), 1, "fixed"),
                ar.TrackedFinding(_finding("style", "low"), 1, "residual", "below fix threshold"),
            ],
        )
        report_path = tmp_path / "reports" / "adversarial_review.json"
        ar.write_report(outcome, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["stop_reason"] == "converged"
        assert payload["rounds_run"] == 2
        assert payload["passed"] is True
        assert payload["findings"][0]["status"] == "fixed"
        assert payload["findings"][1]["reason_unfixed"] == "below fix threshold"


class TestLoadLoopConfig:
    def test_reads_tool_section(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.adversarial-review]\nmax_iterations = 7\nfix_threshold = "critical"\n',
            encoding="utf-8",
        )
        config = ar.load_loop_config(pyproject)
        assert config.max_iterations == 7
        assert config.fix_threshold == "critical"

    def test_missing_section_uses_defaults(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "x"\n', encoding="utf-8")
        config = ar.load_loop_config(pyproject)
        assert config.max_iterations == 5
        assert config.fix_threshold == "high"

    def test_missing_tool_table_uses_defaults(self, tmp_path: Path) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("", encoding="utf-8")
        assert ar.load_loop_config(pyproject).max_iterations == 5


class TestClaudeAgents:
    def test_reviewer_invokes_claude_and_parses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("adversarial_review.shutil.which", _which_claude)
        captured: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.append(command)
            stdout = '[{"identifier": "a", "description": "d", "severity": "high"}]'
            return subprocess.CompletedProcess(command, 0, stdout=stdout)

        monkeypatch.setattr("adversarial_review.subprocess.run", fake_run)
        findings = ar.claude_reviewer(1, [])
        assert findings[0].identifier == "a"
        assert captured[0][0] == "/usr/bin/claude"
        assert captured[0][1] == "-p"

    def test_fixer_passes_findings_in_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("adversarial_review.shutil.which", _which_claude)
        captured: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.append(command)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr("adversarial_review.subprocess.run", fake_run)
        ar.claude_fixer([_finding("a", "high")])
        assert "--permission-mode" in captured[0]
        assert any('"a"' in part for part in captured[0])

    def test_missing_claude_binary_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("adversarial_review.shutil.which", _which_missing)
        with pytest.raises(SystemExit, match="claude CLI not found"):
            ar.claude_reviewer(1, [])

    def test_gates_pass_maps_exit_codes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("adversarial_review.gatelib.run_gates", _fake_run_gates(0))
        assert ar.gates_pass() is True
        monkeypatch.setattr("adversarial_review.gatelib.run_gates", _fake_run_gates(1))
        assert ar.gates_pass() is False


class TestMain:
    def _pyproject(self, tmp_path: Path) -> Path:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            '[tool.adversarial-review]\nreport_path = "reports/review.json"\n',
            encoding="utf-8",
        )
        return pyproject

    def test_pass_writes_report_and_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyproject = self._pyproject(tmp_path)
        monkeypatch.setattr(ar, "claude_reviewer", _ScriptedReviewer([[]]))
        exit_code = ar.main(["--pyproject", str(pyproject)])
        assert exit_code == 0
        payload = json.loads((tmp_path / "reports" / "review.json").read_text(encoding="utf-8"))
        assert payload["stop_reason"] == "converged"

    def test_escalation_returns_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pyproject = self._pyproject(tmp_path)
        finding = _finding("flip-flop", "critical")
        monkeypatch.setattr(ar, "claude_reviewer", _ScriptedReviewer([[finding], [finding]]))
        monkeypatch.setattr(ar, "claude_fixer", _no_fix)
        monkeypatch.setattr(ar, "gates_pass", _gates_green)
        assert ar.main(["--pyproject", str(pyproject)]) == 1
