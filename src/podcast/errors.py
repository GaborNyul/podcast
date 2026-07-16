# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Typed exception hierarchy; raised deep, rendered once at the CLI boundary."""

from typing import ClassVar


class PodcastError(Exception):
    """Base for all podcast tool errors; `exit_code` reaches the shell."""

    exit_code: ClassVar[int] = 1


class ConfigError(PodcastError):
    """Invalid or unreadable configuration."""

    exit_code: ClassVar[int] = 2


class IngestError(PodcastError):
    """A source document could not be loaded or converted."""

    exit_code: ClassVar[int] = 3


class ProviderError(PodcastError):
    """An LLM provider failed (auth, transport, or malformed response)."""

    exit_code: ClassVar[int] = 4


class ScriptError(PodcastError):
    """Script generation produced unusable output."""

    exit_code: ClassVar[int] = 5


class TTSError(PodcastError):
    """A TTS engine is unavailable or failed to synthesize."""

    exit_code: ClassVar[int] = 6


class AudioError(PodcastError):
    """Audio assembly (ffmpeg) failed."""

    exit_code: ClassVar[int] = 7
