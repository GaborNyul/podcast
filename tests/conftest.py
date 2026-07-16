# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Shared fixtures; also makes the standalone gate scripts importable."""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
# Credentials for tests that talk to real services (e.g. HF_TOKEN for the
# integration-marked model downloads); never overrides real env vars.
load_dotenv(REPO_ROOT / ".env", override=False)
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# mutmut runs the whole suite several times in one process (stats, clean run,
# per-mutant). That trips Hypothesis' differing_executors health check on the
# method-based property tests — a false positive here, not a test defect. Only
# tests without their own @settings inherit this; the rest opt in explicitly.
if os.environ.get("MUTANT_UNDER_TEST"):
    from hypothesis import HealthCheck, settings

    settings.register_profile("mutmut", suppress_health_check=[HealthCheck.differing_executors])
    settings.load_profile("mutmut")


@pytest.fixture(autouse=True)
def _mutmut_absolute_source_paths(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep mutation testing working across tests that change the working dir.

    mutmut's trampoline calls ``record_trampoline_hit`` on every mutated function,
    which does ``Path(p).resolve(strict=True)`` over the configured (relative)
    ``source_paths``. Tests that ``chdir`` elsewhere (e.g. ``isolated_env``) make
    that strict resolve raise ``FileNotFoundError``, aborting the mutmut stats run.
    Anchoring the paths to the mutants root (``REPO_ROOT`` here) makes the resolve
    cwd-independent. Restored per-test, so mutmut's own bookkeeping is untouched.
    No-op outside a mutation run.
    """
    if not os.environ.get("MUTANT_UNDER_TEST"):
        return
    try:
        from mutmut.configuration import Config
    except ImportError:
        return
    config = Config.get()
    monkeypatch.setattr(
        config, "source_paths", [REPO_ROOT / p for p in config.source_paths], raising=False
    )


def write_minimal_docx(path: Path, text: str) -> Path:
    """A smallest-possible OOXML document that mammoth/markitdown can read."""
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '<Default Extension="xml" ContentType="application/xml"/>\n'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
        '2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        "</Relationships>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return path


def write_minimal_pdf(path: Path, text: str) -> Path:
    """A smallest-possible PDF with one line of Helvetica text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    body = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        f"4 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj\n"
        "5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "trailer << /Root 1 0 R >>\n"
        "%%EOF"
    )
    path.write_bytes(body.encode("latin-1"))
    return path


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty cwd + isolated XDG dirs + no PODCAST_* env leaking into config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    for name in [key for key in os.environ if key.startswith("PODCAST_")]:
        monkeypatch.delenv(name)
    return tmp_path
