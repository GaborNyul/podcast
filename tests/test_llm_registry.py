# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.llm.registry."""

import httpx
import pytest

from podcast.config import AppConfig, LLMSettings
from podcast.errors import ProviderError
from podcast.llm import registry
from podcast.llm.anthropic import AnthropicProvider
from podcast.llm.fake import FakeProvider
from podcast.llm.openai_compat import OpenAICompatProvider


def _config(**llm: object) -> AppConfig:
    return AppConfig(llm=LLMSettings.model_validate(llm))


class TestAvailableProviders:
    def test_lists_all_factories_sorted(self) -> None:
        assert registry.available_providers() == [
            "anthropic",
            "copilot",
            "fake",
            "ollama",
            "ollama-cloud",
            "openai",
        ]


class TestCreateProvider:
    def test_unknown_provider_names_available(self) -> None:
        with pytest.raises(ProviderError, match="unknown LLM provider 'psychic'"):
            registry.create_provider(_config(provider="psychic"))

    def test_fake_provider(self) -> None:
        provider = registry.create_provider(_config(provider="fake"))
        assert isinstance(provider, FakeProvider)

    def test_ollama_defaults(self) -> None:
        provider = registry.create_provider(_config(provider="ollama"))
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.model == registry.DEFAULT_MODELS["ollama"]

    def test_explicit_model_wins(self) -> None:
        provider = registry.create_provider(_config(provider="ollama", model="llama3:70b"))
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.model == "llama3:70b"

    def test_ollama_cloud_requires_api_key(self) -> None:
        with pytest.raises(ProviderError, match="PODCAST_LLM__API_KEY"):
            registry.create_provider(_config(provider="ollama-cloud"))

    def test_ollama_cloud_with_key(self) -> None:
        provider = registry.create_provider(_config(provider="ollama-cloud", api_key="ok-123"))
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.name == "ollama-cloud"

    def test_openai_requires_api_key(self) -> None:
        with pytest.raises(ProviderError, match="openai requires an API key"):
            registry.create_provider(_config(provider="openai"))

    def test_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Client:
            def __init__(self, *, api_key: str | None = None, timeout: float = 0) -> None:
                self.api_key = api_key
                self.timeout = timeout

        monkeypatch.setattr("podcast.llm.anthropic.anthropic.Anthropic", _Client)
        provider = registry.create_provider(_config(provider="anthropic"))
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-opus-4-8"

    def test_copilot_exchanges_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_obtain(client: httpx.Client, **_kwargs: object) -> str:
            assert isinstance(client, httpx.Client)
            return "cop_bearer"

        monkeypatch.setattr("podcast.llm.registry.copilot_auth.obtain_bearer", fake_obtain)
        provider = registry.create_provider(_config(provider="copilot"))
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.name == "copilot"
        assert provider.model == registry.DEFAULT_MODELS["copilot"]


class TestAnnounce:
    def test_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        registry.announce_to_stderr("visit github.com")
        assert "visit github.com" in capsys.readouterr().err
