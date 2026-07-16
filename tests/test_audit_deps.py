# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for scripts/audit_deps.py."""

import subprocess
from collections.abc import Sequence

import pytest

import audit_deps


def _which_uv(_name: str) -> str:
    return "/usr/bin/uv"


def _which_missing(_name: str) -> None:
    return None


def _completed(
    command: Sequence[str], returncode: int, stdout: str = "", stderr: str = ""
) -> "subprocess.CompletedProcess[str]":
    return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)


class TestRunCommand:
    def test_returns_completed_process(self) -> None:
        import sys

        result = audit_deps.run_command([sys.executable, "-c", "print('hi')"])
        assert result.returncode == 0
        assert result.stdout.strip() == "hi"


class TestMain:
    def test_clean_audit_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_uv)
        commands: list[list[str]] = []

        def runner(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
            commands.append(list(command))
            if "export" in command:
                return _completed(command, 0, stdout="typer==0.16.0\n")
            return _completed(command, 0, stdout="No known vulnerabilities found")

        assert audit_deps.main(runner) == 0
        assert commands[0][:2] == ["/usr/bin/uv", "export"]
        assert "--no-deps" in commands[1]
        assert "No known vulnerabilities" in capsys.readouterr().out

    def test_requirements_file_carries_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_uv)
        seen: list[str] = []

        def runner(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
            if "export" in command:
                return _completed(command, 0, stdout="httpx==0.28.0\n")
            requirements = command[command.index("-r") + 1]
            from pathlib import Path

            seen.append(Path(requirements).read_text(encoding="utf-8"))
            return _completed(command, 0)

        assert audit_deps.main(runner) == 0
        assert seen == ["httpx==0.28.0\n"]

    def test_vulnerable_dependency_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_uv)

        def runner(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
            if "export" in command:
                return _completed(command, 0, stdout="badpkg==1.0\n")
            return _completed(command, 1, stdout="found 1 known vulnerability")

        assert audit_deps.main(runner) == 1

    def test_export_failure_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_uv)

        def runner(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
            return _completed(command, 2, stderr="lockfile out of date")

        assert audit_deps.main(runner) == 1
        assert "lockfile out of date" in capsys.readouterr().out

    def test_missing_uv_fails(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_missing)
        assert audit_deps.main() == 1
        assert "uv not found" in capsys.readouterr().out

    def test_stderr_printed_when_stdout_empty(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("audit_deps.shutil.which", _which_uv)

        def runner(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
            if "export" in command:
                return _completed(command, 0, stdout="pkg==1.0\n")
            return _completed(command, 0, stderr="warning: audited from stderr")

        assert audit_deps.main(runner) == 0
        assert "audited from stderr" in capsys.readouterr().out
