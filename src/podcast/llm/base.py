# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Chat provider protocol and message types."""

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """One chat turn; providers translate this to their native wire format."""

    role: Literal["system", "user", "assistant"]
    content: str


def system(content: str) -> ChatMessage:
    return ChatMessage(role="system", content=content)


def user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def assistant(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


@runtime_checkable
class ChatProvider(Protocol):
    """Minimal synchronous chat interface every provider adapts to."""

    name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        """Return the assistant's text, honoring `json_schema` natively when supported."""
        ...
