"""Shared fixtures; also makes the standalone gate scripts importable."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty cwd + isolated XDG dirs + no PODCAST_* env leaking into config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    for name in [key for key in os.environ if key.startswith("PODCAST_")]:
        monkeypatch.delenv(name)
    return tmp_path
