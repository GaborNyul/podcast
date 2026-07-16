# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for scripts/check_commit_msg.py."""

from pathlib import Path

import pytest

import check_commit_msg

CLEAN = """feat(tts): validate [tts.voices] overrides against the active engine

An override naming another engine's voice now fails at resolve time.
"""


class TestStripGitNoise:
    def test_drops_comment_lines(self) -> None:
        message = "feat: thing\n# Please enter the commit message\n\nbody\n"
        assert check_commit_msg.strip_git_noise(message) == "feat: thing\n\nbody"

    def test_truncates_at_verbose_scissors(self) -> None:
        message = (
            "feat: thing\n"
            "# ------------------------ >8 ------------------------\n"
            "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\n"
        )
        assert check_commit_msg.strip_git_noise(message) == "feat: thing"

    def test_keeps_ordinary_body(self) -> None:
        assert check_commit_msg.strip_git_noise("a\nb") == "a\nb"


class TestFindSignatures:
    def test_clean_message_has_none(self) -> None:
        assert check_commit_msg.find_signatures(CLEAN) == []

    def test_detects_co_authored_by_claude(self) -> None:
        message = f"{CLEAN}\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
        assert check_commit_msg.find_signatures(message) == [
            "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
        ]

    def test_detects_claude_session_trailer(self) -> None:
        message = f"{CLEAN}\nClaude-Session: https://claude.ai/code/session_01WgRcAa\n"
        assert check_commit_msg.find_signatures(message) == [
            "Claude-Session: https://claude.ai/code/session_01WgRcAa"
        ]

    def test_detects_generated_with_footer(self) -> None:
        message = f"{CLEAN}\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n"
        assert len(check_commit_msg.find_signatures(message)) == 1

    def test_detects_bare_anthropic_noreply_address(self) -> None:
        message = f"{CLEAN}\nSigned-off-by: Someone <noreply@anthropic.com>\n"
        assert len(check_commit_msg.find_signatures(message)) == 1

    def test_overlapping_patterns_report_line_once(self) -> None:
        # Matches both the Co-Authored-By and the noreply@anthropic.com pattern.
        message = f"{CLEAN}\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
        assert len(check_commit_msg.find_signatures(message)) == 1

    def test_reports_every_offending_line(self) -> None:
        message = (
            f"{CLEAN}\n"
            "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\n"
            "Claude-Session: https://claude.ai/code/session_01WgRcAa\n"
        )
        assert len(check_commit_msg.find_signatures(message)) == 2

    def test_anthropic_scope_is_not_a_signature(self) -> None:
        # e772a53 is a real commit: it discusses the Anthropic API and must pass.
        message = (
            "fix(anthropic): rewrite JSON schemas for structured-outputs compliance\n\n"
            "Anthropic's output_config.format rejects object schemas without\n"
            "additionalProperties: false.\n"
        )
        assert check_commit_msg.find_signatures(message) == []

    def test_prose_about_claude_is_not_a_signature(self) -> None:
        message = "docs: note that Claude Code adds trailers by default\n"
        assert check_commit_msg.find_signatures(message) == []

    def test_commented_out_signature_is_ignored(self) -> None:
        message = f"{CLEAN}\n# Co-Authored-By: Claude <noreply@anthropic.com>\n"
        assert check_commit_msg.find_signatures(message) == []


class TestMain:
    def test_clean_message_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(CLEAN, encoding="utf-8")
        assert check_commit_msg.main([str(path)]) == 0

    def test_signed_message_fails_and_quotes_the_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(f"{CLEAN}\nCo-Authored-By: Claude Fable 5 <x@y.z>\n", encoding="utf-8")
        assert check_commit_msg.main([str(path)]) == 1
        out = capsys.readouterr().out
        assert "Co-Authored-By: Claude Fable 5" in out
        assert "python_development_standards_v3.md" in out

    def test_missing_argument_fails(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert check_commit_msg.main([]) == 1
        assert "usage:" in capsys.readouterr().out

    def test_falls_back_to_sys_argv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "COMMIT_EDITMSG"
        path.write_text(CLEAN, encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["check_commit_msg.py", str(path)])
        assert check_commit_msg.main() == 0
