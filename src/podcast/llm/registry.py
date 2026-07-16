# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Provider registry: config name → constructed ChatProvider."""

import sys
from collections.abc import Callable

import httpx

from podcast.config import AppConfig
from podcast.errors import ProviderError
from podcast.llm import copilot_auth
from podcast.llm.anthropic import AnthropicProvider
from podcast.llm.base import ChatProvider
from podcast.llm.fake import FakeProvider
from podcast.llm.openai_compat import OpenAICompatProvider

type ProviderFactory = Callable[[AppConfig], ChatProvider]

DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3:30b-a3b-instruct-2507",
    "ollama-cloud": "qwen3-coder:480b-cloud",
    "openai": "gpt-5",
    "copilot": "gpt-4o",
    "anthropic": "claude-opus-4-8",
}

OLLAMA_LOCAL_URL = "http://localhost:11434/v1"
OLLAMA_CLOUD_URL = "https://ollama.com/v1"
OPENAI_URL = "https://api.openai.com/v1"
COPILOT_URL = "https://api.githubcopilot.com"


def _model_for(config: AppConfig, provider: str) -> str:
    return config.llm.model if config.llm.model is not None else DEFAULT_MODELS[provider]


def _require_api_key(config: AppConfig, provider: str) -> str:
    if config.llm.api_key is None:
        raise ProviderError(
            f"{provider} requires an API key; set PODCAST_LLM__API_KEY or [llm].api_key"
        )
    return config.llm.api_key


def _ollama(config: AppConfig) -> ChatProvider:
    return OpenAICompatProvider(
        "ollama",
        base_url=config.llm.base_url or OLLAMA_LOCAL_URL,
        model=_model_for(config, "ollama"),
        api_key=config.llm.api_key,
        timeout_seconds=config.llm.timeout_seconds,
    )


def _ollama_cloud(config: AppConfig) -> ChatProvider:
    return OpenAICompatProvider(
        "ollama-cloud",
        base_url=config.llm.base_url or OLLAMA_CLOUD_URL,
        model=_model_for(config, "ollama-cloud"),
        api_key=_require_api_key(config, "ollama-cloud"),
        timeout_seconds=config.llm.timeout_seconds,
    )


def _openai(config: AppConfig) -> ChatProvider:
    return OpenAICompatProvider(
        "openai",
        base_url=config.llm.base_url or OPENAI_URL,
        model=_model_for(config, "openai"),
        api_key=_require_api_key(config, "openai"),
        timeout_seconds=config.llm.timeout_seconds,
    )


def announce_to_stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _copilot(config: AppConfig) -> ChatProvider:
    with httpx.Client(timeout=30.0) as auth_client:
        bearer = copilot_auth.obtain_bearer(auth_client, announce=announce_to_stderr)
    return OpenAICompatProvider(
        "copilot",
        base_url=config.llm.base_url or COPILOT_URL,
        model=_model_for(config, "copilot"),
        api_key=bearer,
        timeout_seconds=config.llm.timeout_seconds,
        extra_headers={
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.99.0",
        },
        supports_json_schema=False,
    )


def _anthropic(config: AppConfig) -> ChatProvider:
    return AnthropicProvider(
        model=_model_for(config, "anthropic"),
        api_key=config.llm.api_key,
        timeout_seconds=config.llm.timeout_seconds,
    )


def _fake(_config: AppConfig) -> ChatProvider:
    return FakeProvider()


FACTORIES: dict[str, ProviderFactory] = {
    "ollama": _ollama,
    "ollama-cloud": _ollama_cloud,
    "openai": _openai,
    "copilot": _copilot,
    "anthropic": _anthropic,
    "fake": _fake,
}


def available_providers() -> list[str]:
    return sorted(FACTORIES)


def create_provider(config: AppConfig) -> ChatProvider:
    """Build the configured provider; unknown names fail with the available list."""
    factory = FACTORIES.get(config.llm.provider)
    if factory is None:
        known = ", ".join(available_providers())
        raise ProviderError(f"unknown LLM provider {config.llm.provider!r} (available: {known})")
    return factory(config)
