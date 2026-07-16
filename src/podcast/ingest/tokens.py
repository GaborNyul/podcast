# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Token counting with a safety factor; degrades to a word heuristic offline."""

import math
from functools import lru_cache
from typing import TYPE_CHECKING

from podcast.errors import IngestError

if TYPE_CHECKING:
    from tiktoken import Encoding

SAFETY_FACTOR = 1.15
_WORDS_TO_TOKENS = 4 / 3  # rough English words→tokens ratio for the offline fallback


@lru_cache(maxsize=1)
def load_encoder() -> "Encoding | None":
    """The cached o200k_base encoding, or None when unavailable (e.g. offline)."""
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:
        # tiktoken fetches its BPE file over the network on first use; a local-first
        # tool must keep working without it.
        return None


def count_tokens(text: str) -> int:
    """Estimated prompt tokens for `text`, inflated by the safety factor."""
    encoder = load_encoder()
    if encoder is not None:
        raw = len(encoder.encode(text))
    else:
        raw = math.ceil(len(text.split()) * _WORDS_TO_TOKENS)
    return math.ceil(raw * SAFETY_FACTOR)


def assert_fits_context(total_tokens: int, context_window: int, reserve: int = 8192) -> None:
    """Fail early when sources cannot fit the model context with `reserve` left for output."""
    budget = context_window - reserve
    if total_tokens > budget:
        raise IngestError(
            f"sources are ~{total_tokens} tokens but the model context leaves room for "
            f"{budget} (window {context_window} minus {reserve} reserved for output); "
            "drop a source or use a larger-context model"
        )
