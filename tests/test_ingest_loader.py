"""Tests for podcast.ingest.loader."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import write_minimal_docx
from podcast.errors import IngestError
from podcast.ingest import loader
from podcast.ingest.loader import (
    SourceDocument,
    load_document,
    load_documents,
    merged_sources_markdown,
)


def _fallback_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("podcast.ingest.tokens.load_encoder", lambda: None)


class TestLoadDocument:
    def test_plain_text_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fallback_tokens(monkeypatch)
        source = tmp_path / "notes.txt"
        source.write_text("Plain text about ants.", encoding="utf-8")
        document = load_document(source)
        assert document.markdown == "Plain text about ants."
        assert document.title == "notes"
        assert document.tokens >= 1
        assert document.path == str(source)

    def test_markdown_title_from_first_heading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fallback_tokens(monkeypatch)
        source = tmp_path / "doc.md"
        source.write_text("intro line\n\n## Ant Colonies\n\nBody.", encoding="utf-8")
        assert load_document(source).title == "Ant Colonies"

    def test_bare_hash_heading_falls_back_to_stem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fallback_tokens(monkeypatch)
        source = tmp_path / "weird.md"
        source.write_text("#\n\ncontent", encoding="utf-8")
        assert load_document(source).title == "weird"

    def test_docx_dispatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fallback_tokens(monkeypatch)
        source = write_minimal_docx(tmp_path / "brief.docx", "Docx body text.")
        assert "Docx body text." in load_document(source).markdown

    def test_html_dispatch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fallback_tokens(monkeypatch)

        def fake_html(_path: Path) -> str:
            return "# Extracted\n\nFrom html."

        monkeypatch.setattr("podcast.ingest.converters.html_to_markdown", fake_html)
        source = tmp_path / "page.html"
        source.write_text("<html></html>", encoding="utf-8")
        document = load_document(source)
        assert document.title == "Extracted"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError, match="not found"):
            load_document(tmp_path / "ghost.txt")

    def test_unsupported_suffix_lists_supported(self, tmp_path: Path) -> None:
        source = tmp_path / "table.xlsx"
        source.write_text("x", encoding="utf-8")
        with pytest.raises(IngestError, match=r"unsupported source type.*\.pdf"):
            load_document(source)

    def test_suffixless_file_is_unsupported(self, tmp_path: Path) -> None:
        source = tmp_path / "README"
        source.write_text("x", encoding="utf-8")
        with pytest.raises(IngestError, match="unsupported source type"):
            load_document(source)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "empty.txt"
        source.write_text("   \n\t\n", encoding="utf-8")
        with pytest.raises(IngestError, match="no extractable text"):
            load_document(source)

    def test_invalid_utf8_is_replaced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fallback_tokens(monkeypatch)
        source = tmp_path / "mixed.txt"
        source.write_bytes(b"caf\xe9 content")
        assert "content" in load_document(source).markdown

    @settings(
        # differing_executors: a false positive under mutmut, which re-runs this
        # method-based property test across instances in one process.
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
        max_examples=50,
    )
    @given(text=st.text(min_size=1).filter(lambda value: value.strip() != ""))
    def test_arbitrary_text_never_crashes(
        self, text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fallback_tokens(monkeypatch)
        source = tmp_path / "fuzz.txt"
        source.write_text(text, encoding="utf-8")
        document = load_document(source)
        assert document.tokens >= 0
        assert document.title != ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        assert document.markdown == normalized


class TestLoadDocuments:
    def test_loads_all_in_order(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _fallback_tokens(monkeypatch)
        first = tmp_path / "a.txt"
        first.write_text("Alpha body.", encoding="utf-8")
        second = tmp_path / "b.md"
        second.write_text("# Beta\n\nBody.", encoding="utf-8")
        documents = load_documents([first, second])
        assert [document.title for document in documents] == ["a", "Beta"]

    def test_empty_input_raises(self) -> None:
        with pytest.raises(IngestError, match="no source documents"):
            load_documents([])

    def test_one_bad_source_fails_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fallback_tokens(monkeypatch)
        good = tmp_path / "good.txt"
        good.write_text("fine", encoding="utf-8")
        with pytest.raises(IngestError, match="not found"):
            load_documents([good, tmp_path / "missing.txt"])


class TestMergedSourcesMarkdown:
    def _document(self, title: str, body: str) -> SourceDocument:
        return SourceDocument(path=f"sources/{title}.md", title=title, markdown=body, tokens=1)

    def test_labels_and_separates_sources(self) -> None:
        merged = merged_sources_markdown(
            [self._document("First", "aaa"), self._document("Second", "bbb")]
        )
        assert "## Source 1: First" in merged
        assert "## Source 2: Second" in merged
        assert "\n\n---\n\n" in merged
        assert merged.index("aaa") < merged.index("bbb")

    def test_single_source_has_no_separator(self) -> None:
        merged = merged_sources_markdown([self._document("Only", "body")])
        assert "---" not in merged

    def test_supported_suffixes_exposed(self) -> None:
        assert ".pdf" in loader.SUPPORTED_SUFFIXES
        assert ".docx" in loader.SUPPORTED_SUFFIXES
