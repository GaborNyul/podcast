# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Every first-party source file must carry the SPDX AGPL header.

Guards the full swept scope (DP-2): the shipped package plus tests and dev
scripts. A new file in any of these roots without the header fails this test.
"""

from pathlib import Path

_MARKER = "# SPDX-License-Identifier: AGPL-3.0-or-later"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROOTS = ("src/podcast", "tests", "scripts")


def _first_party_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _ROOTS:
        files.extend((_REPO_ROOT / root).rglob("*.py"))
    return sorted(files)


def test_every_first_party_file_has_spdx_header() -> None:
    missing = [
        str(path.relative_to(_REPO_ROOT))
        for path in _first_party_py_files()
        if _MARKER not in path.read_text(encoding="utf-8")
    ]
    assert missing == [], f"missing SPDX header in: {missing}"
