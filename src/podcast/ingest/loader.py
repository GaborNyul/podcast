# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Load source files into labeled markdown documents."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from podcast.errors import IngestError
from podcast.ingest import tokens

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx"})


class SourceDocument(BaseModel):
    """One ingested source: markdown content plus provenance and size metadata."""

    path: str
    title: str
    markdown: str
    tokens: int


def _extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading
    return fallback


def _to_markdown(path: Path) -> str:
    from podcast.ingest import converters

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return converters.html_to_markdown(path)
    return converters.convert_with_markitdown(path)


def load_document(path: Path) -> SourceDocument:
    """Ingest one file; raises IngestError for missing/unsupported/empty inputs."""
    if not path.is_file():
        raise IngestError(f"source file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise IngestError(f"unsupported source type {suffix or path.name!r} ({supported})")
    markdown = _to_markdown(path).strip()
    if not markdown:
        raise IngestError(f"no extractable text in {path}")
    return SourceDocument(
        path=str(path),
        title=_extract_title(markdown, path.stem),
        markdown=markdown,
        tokens=tokens.count_tokens(markdown),
    )


def load_documents(paths: Sequence[Path]) -> list[SourceDocument]:
    """Ingest all sources for an episode; empty input is an error."""
    if not paths:
        raise IngestError("no source documents given")
    return [load_document(path) for path in paths]


def merged_sources_markdown(documents: Sequence[SourceDocument]) -> str:
    """All sources as one grounding text with per-source labels the LLM can cite."""
    sections = [
        f"## Source {index}: {document.title}\n\n{document.markdown}"
        for index, document in enumerate(documents, start=1)
    ]
    return "\n\n---\n\n".join(sections)
