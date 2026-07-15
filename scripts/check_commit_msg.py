"""Commit-message gate: reject AI assistant signatures.

Standards v3 forbids assistant signatures in commit messages (see the "Git Commit
Rules" section of docs/python_development_standards_v3.md). The rule predates this
gate; two commits reached main carrying `Co-Authored-By: Claude` trailers anyway,
so it is enforced here at commit time rather than left to prose.

Patterns match signature *forms*, never the bare words "claude" or "anthropic":
commits legitimately discuss the Anthropic API (the `fix(anthropic):` scope, for
one), and rejecting those would make the gate a nuisance instead of a guardrail.
"""

import re
import sys
from collections.abc import Sequence
from pathlib import Path

SIGNATURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Co-Authored-By:.*\bClaude\b", re.IGNORECASE),
    re.compile(r"^\s*Claude-Session:", re.IGNORECASE),
    re.compile(r"Generated with \[?Claude Code", re.IGNORECASE),
    re.compile(r"noreply@anthropic\.com", re.IGNORECASE),
)


def strip_git_noise(message: str) -> str:
    """Drop comment lines, and anything below a `commit --verbose` scissors line.

    git strips these itself after the hook runs, so a signature quoted in the
    commented help text — or in a verbose diff — is not a real signature.
    """
    kept: list[str] = []
    for line in message.splitlines():
        if line.startswith("#"):
            if ">8" in line:
                break
            continue
        kept.append(line)
    return "\n".join(kept)


def find_signatures(message: str) -> list[str]:
    """Return the message's assistant-signature lines, stripped, in order."""
    return [
        line.strip()
        for line in strip_git_noise(message).splitlines()
        if any(pattern.search(line) for pattern in SIGNATURE_PATTERNS)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: check_commit_msg.py <commit-msg-file>")
        return 1
    offenders = find_signatures(Path(args[0]).read_text(encoding="utf-8"))
    if not offenders:
        return 0
    print("Assistant signature in commit message — rejected:")
    for line in offenders:
        print(f"    {line}")
    print("\nStandards v3 forbids these: docs/python_development_standards_v3.md")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
