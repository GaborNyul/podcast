# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Format converters: markitdown backbone, trafilatura for article HTML."""

from pathlib import Path

from podcast.errors import IngestError


def convert_with_markitdown(path: Path) -> str:
    """Convert pdf/docx/html to markdown text via markitdown."""
    from markitdown import MarkItDown

    converter = MarkItDown(enable_plugins=False)
    try:
        result = converter.convert(str(path))
    except Exception as exc:
        raise IngestError(f"cannot convert {path}: {exc}") from exc
    return result.text_content


def html_to_markdown(path: Path) -> str:
    """Extract article HTML with trafilatura; fall back to markitdown for app-like pages."""
    import trafilatura

    html = path.read_text(encoding="utf-8", errors="replace")
    extracted = trafilatura.extract(
        html, output_format="markdown", include_links=False, include_formatting=True
    )
    if extracted:
        return extracted
    return convert_with_markitdown(path)
