# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Every shipped source file must carry the SPDX AGPL header."""

from pathlib import Path

import podcast

_MARKER = "# SPDX-License-Identifier: AGPL-3.0-or-later"
_PACKAGE_ROOT = Path(podcast.__file__).parent


def test_every_package_file_has_spdx_header() -> None:
    missing = [
        str(path.relative_to(_PACKAGE_ROOT))
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if _MARKER not in path.read_text(encoding="utf-8")
    ]
    assert missing == [], f"missing SPDX header in: {missing}"
