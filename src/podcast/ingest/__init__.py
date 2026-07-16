# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
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
