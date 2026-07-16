# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.ingest.converters."""

from pathlib import Path

import pytest

from conftest import write_minimal_docx, write_minimal_pdf
from podcast.errors import IngestError
from podcast.ingest import converters

ARTICLE_HTML = """<!DOCTYPE html><html><head><title>Solar Sails</title></head><body>
<nav><a href="/">Home</a><a href="/about">About</a></nav>
<article>
<h1>Solar Sails Explained</h1>
<p>Solar sails use radiation pressure from sunlight to propel spacecraft without fuel.
The concept dates back over a century and has now been demonstrated in orbit.</p>
<p>A sail the size of a football field receives only a few newtons of force, yet over
months of continuous acceleration it can reach speeds impossible for chemical rockets.</p>
<p>Missions like IKAROS and LightSail proved the principle works in practice, steering
by tilting the sail relative to the sun.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>"""


class TestConvertWithMarkitdown:
    def test_docx_extracts_text(self, tmp_path: Path) -> None:
        source = write_minimal_docx(tmp_path / "note.docx", "Hello from docx fixture.")
        assert "Hello from docx fixture." in converters.convert_with_markitdown(source)

    def test_pdf_extracts_text(self, tmp_path: Path) -> None:
        source = write_minimal_pdf(tmp_path / "note.pdf", "Hello from pdf fixture.")
        assert "Hello from pdf fixture." in converters.convert_with_markitdown(source)

    def test_corrupt_file_raises_ingest_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "broken.docx"
        corrupt.write_bytes(bytes(range(256)) * 4)
        with pytest.raises(IngestError, match="cannot convert"):
            converters.convert_with_markitdown(corrupt)


class TestHtmlToMarkdown:
    def test_article_html_extracts_via_trafilatura(self, tmp_path: Path) -> None:
        page = tmp_path / "article.html"
        page.write_text(ARTICLE_HTML, encoding="utf-8")
        markdown = converters.html_to_markdown(page)
        assert "# Solar Sails Explained" in markdown
        assert "Copyright 2026" not in markdown  # boilerplate stripped

    def test_empty_page_falls_back_to_markitdown(self, tmp_path: Path) -> None:
        page = tmp_path / "empty.html"
        page.write_text("<html><body></body></html>", encoding="utf-8")
        assert converters.html_to_markdown(page) == ""

    def test_invalid_bytes_are_replaced_not_fatal(self, tmp_path: Path) -> None:
        page = tmp_path / "latin.html"
        page.write_bytes(b"<html><body><p>caf\xe9 " + b"story " * 50 + b"</p></body></html>")
        markdown = converters.html_to_markdown(page)
        assert "story" in markdown
