"""Document ingestion: files in, labeled markdown sources out."""

from podcast.ingest.loader import (
    SUPPORTED_SUFFIXES,
    SourceDocument,
    load_document,
    load_documents,
    merged_sources_markdown,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "SourceDocument",
    "load_document",
    "load_documents",
    "merged_sources_markdown",
]
