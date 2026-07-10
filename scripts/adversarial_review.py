"""Bounded adversarial review-fix loop (standards v3, Automated Adversarial Code Review)."""

import argparse
import json
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

import gatelib

type Severity = Literal["critical", "high", "medium", "low"]
type FindingStatus = Literal["open", "fixed", "residual", "escalated"]
type StopReason = Literal["converged", "cap-reached", "oscillation"]

_SEVERITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class ReviewFinding(BaseModel):
    """One reviewer finding; `identifier` must stay stable for the same fault."""

    identifier: str
    description: str
    severity: Severity


class LoopConfig(BaseModel):
    """[tool.adversarial-review] in pyproject.toml."""

    max_iterations: int = 5
    fix_threshold: Severity = "high"
    report_path: Path = Path("reports/adversarial_review.json")


@dataclass
class TrackedFinding:
    """A finding's lifecycle across rounds; severity re-grades are ignored (anti-gaming)."""

    finding: ReviewFinding
    first_round: int
    status: FindingStatus = "open"
    reason_unfixed: str = ""


@dataclass(frozen=True)
class LoopOutcome:
    stop_reason: StopReason
    rounds_run: int
    findings: list[TrackedFinding]

    @property
    def passed(self) -> bool:
        """Auto-accept only when nothing above the threshold remains unfixed."""
        return all(item.status in {"fixed", "residual"} for item in self.findings)


type Reviewer = Callable[[int, Sequence[ReviewFinding]], list[ReviewFinding]]
type Fixer = Callable[[Sequence[ReviewFinding]], None]
type GateCheck = Callable[[], bool]


def must_fix(finding: ReviewFinding, threshold: Severity) -> bool:
    return _SEVERITY_RANK[finding.severity] <= _SEVERITY_RANK[threshold]


def run_loop(config: LoopConfig, reviewer: Reviewer, fixer: Fixer, gates: GateCheck) -> LoopOutcome:
    """Iterate review→fix until convergence, the round cap, or oscillation."""
    tracked: dict[str, TrackedFinding] = {}
    rounds_run = 0
    stop_reason: StopReason = "cap-reached"

    while rounds_run < config.max_iterations:
        rounds_run += 1
        known = [item.finding for item in tracked.values()]
        reported = reviewer(rounds_run, known)

        oscillated = False
        new_must_fix = 0
        for finding in reported:
            existing = tracked.get(finding.identifier)
            if existing is None:
                tracked[finding.identifier] = TrackedFinding(finding, rounds_run)
                if must_fix(finding, config.fix_threshold):
                    new_must_fix += 1
            elif existing.status == "fixed":
                existing.status = "escalated"
                existing.reason_unfixed = (
                    f"resurfaced in round {rounds_run} after being fixed: "
                    "fix-A-breaks-B thrash, needs human judgment"
                )
                oscillated = True
        if oscillated:
            stop_reason = "oscillation"
            break

        open_must_fix = [
            item.finding
            for item in tracked.values()
            if item.status == "open" and must_fix(item.finding, config.fix_threshold)
        ]
        if new_must_fix == 0 and not open_must_fix:
            stop_reason = "converged"
            break

        fixer(open_must_fix)
        if gates():
            for finding in open_must_fix:
                tracked[finding.identifier].status = "fixed"
        else:
            print(f"round {rounds_run}: fixes broke a quality gate — no progress")

    for item in tracked.values():
        if item.status != "open":
            continue
        if must_fix(item.finding, config.fix_threshold):
            item.status = "escalated"
            item.reason_unfixed = f"still open at stop ({stop_reason}); requires human sign-off"
        else:
            item.status = "residual"
            item.reason_unfixed = "below fix threshold; logged, not fixed"

    return LoopOutcome(stop_reason, rounds_run, list(tracked.values()))


def write_report(outcome: LoopOutcome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stop_reason": outcome.stop_reason,
        "rounds_run": outcome.rounds_run,
        "passed": outcome.passed,
        "findings": [
            {
                "identifier": item.finding.identifier,
                "description": item.finding.description,
                "severity": item.finding.severity,
                "status": item.status,
                "first_round": item.first_round,
                "reason_unfixed": item.reason_unfixed,
            }
            for item in outcome.findings
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_loop_config(pyproject: Path) -> LoopConfig:
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    tool = data.get("tool")
    section: object = {}
    if isinstance(tool, dict):
        section = cast("dict[str, object]", tool).get("adversarial-review", {})
    return LoopConfig.model_validate(section)


def parse_findings(raw: str) -> list[ReviewFinding]:
    """Extract the JSON array from reviewer output that may carry surrounding prose."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("reviewer output contains no JSON array")
    adapter: TypeAdapter[list[ReviewFinding]] = TypeAdapter(list[ReviewFinding])
    try:
        return adapter.validate_json(raw[start : end + 1])
    except ValidationError as exc:
        raise ValueError(f"reviewer output failed validation: {exc}") from exc


_REVIEWER_PROMPT = """You are an adversarial code reviewer for the repository in the \
current directory. Your job is to actively try to break the code and surface faults — \
correctness bugs, security issues, broken contracts, misleading tests — NOT to confirm \
that it works. Review the current working tree. This is round {round}.
Previously reported findings (report again ONLY if still present; keep identifiers \
stable): {known}
Output ONLY a JSON array (no prose, no fences). Each element: {{"identifier": \
"stable-kebab-slug", "description": "...", "severity": "critical|high|medium|low"}}. \
Output [] if you find nothing new."""

_FIXER_PROMPT = """You are the fixer agent in an adversarial review loop. Apply \
minimal, correct fixes in the current working tree for exactly these findings, and \
nothing else: {findings}
Do not refactor unrelated code. Do not change tests to hide a fault."""


def _claude_binary() -> str:
    exe = shutil.which("claude")
    if exe is None:
        raise SystemExit("claude CLI not found on PATH; cannot run agent-driven review")
    return exe


def claude_reviewer(round_no: int, known: Sequence[ReviewFinding]) -> list[ReviewFinding]:
    prompt = _REVIEWER_PROMPT.format(
        round=round_no,
        known=json.dumps([item.model_dump() for item in known]),
    )
    # S603: fixed argv, binary resolved via shutil.which.
    proc = subprocess.run(  # noqa: S603
        [_claude_binary(), "-p", prompt],
        capture_output=True,
        text=True,
        check=True,
        cwd=gatelib.REPO_ROOT,
    )
    return parse_findings(proc.stdout)


def claude_fixer(findings: Sequence[ReviewFinding]) -> None:
    prompt = _FIXER_PROMPT.format(findings=json.dumps([item.model_dump() for item in findings]))
    # S603: fixed argv, binary resolved via shutil.which.
    subprocess.run(  # noqa: S603
        [_claude_binary(), "--permission-mode", "acceptEdits", "-p", prompt],
        check=True,
        cwd=gatelib.REPO_ROOT,
    )


def gates_pass() -> bool:
    results = gatelib.run_gates(gatelib.default_gates())
    return all(code == 0 for _, code in results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=gatelib.REPO_ROOT / "pyproject.toml",
        help="pyproject.toml carrying [tool.adversarial-review]",
    )
    args = parser.parse_args(argv)
    config = load_loop_config(args.pyproject)
    outcome = run_loop(config, claude_reviewer, claude_fixer, gates_pass)
    report_path = args.pyproject.parent / config.report_path
    write_report(outcome, report_path)
    print(
        f"adversarial review: {outcome.stop_reason} after {outcome.rounds_run} round(s); "
        f"{'pass' if outcome.passed else 'FLAG FOR HUMAN SIGN-OFF'} → {report_path}"
    )
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
