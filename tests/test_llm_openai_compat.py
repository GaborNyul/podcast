# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.llm.openai_compat (respx-mocked HTTP)."""

import json

import httpx
import pytest
import respx

from podcast.errors import ProviderError
from podcast.llm.base import user
from podcast.llm.openai_compat import OpenAICompatProvider

BASE_URL = "http://llm.test/v1"


def _provider(**overrides: object) -> OpenAICompatProvider:
    kwargs: dict[str, object] = {
        "base_url": BASE_URL,
        "model": "test-model",
        "api_key": "sk-key",  # pragma: allowlist secret
        "timeout_seconds": 5.0,
    }
    kwargs.update(overrides)
    return OpenAICompatProvider("testprov", **kwargs)  # type: ignore[arg-type]


def _ok_response(content: str | None = "hello") -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


class TestComplete:
    def test_returns_assistant_text(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response("the answer")
        )
        result = _provider().complete([user("q")], temperature=0.4)
        assert result == "the answer"
        payload = json.loads(route.calls.last.request.content)
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.4
        assert payload["messages"] == [{"role": "user", "content": "q"}]
        assert "response_format" not in payload

    def test_sends_bearer_and_extra_headers(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(return_value=_ok_response())
        provider = _provider(extra_headers={"X-Custom": "yes"})
        provider.complete([user("q")], temperature=0.1)
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer sk-key"
        assert request.headers["X-Custom"] == "yes"

    def test_json_schema_becomes_response_format(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response("{}")
        )
        schema: dict[str, object] = {"type": "object", "properties": {}}
        _provider().complete([user("q")], temperature=0.1, json_schema=schema)
        payload = json.loads(route.calls.last.request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == schema
        assert payload["response_format"]["json_schema"]["strict"] is True

    def test_schema_omitted_when_unsupported(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=_ok_response("{}")
        )
        provider = _provider(supports_json_schema=False)
        provider.complete([user("q")], temperature=0.1, json_schema={"type": "object"})
        payload = json.loads(route.calls.last.request.content)
        assert "response_format" not in payload

    def test_no_auth_header_without_api_key(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(return_value=_ok_response())
        provider = _provider(api_key=None)
        provider.complete([user("q")], temperature=0.1)
        assert "Authorization" not in route.calls.last.request.headers

    def test_http_error_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        with pytest.raises(ProviderError, match="HTTP 401"):
            _provider().complete([user("q")], temperature=0.1)

    def test_transport_error_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(ProviderError, match="cannot reach testprov"):
            _provider().complete([user("q")], temperature=0.1)

    def test_malformed_body_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, text="not json")
        )
        with pytest.raises(ProviderError, match="malformed response"):
            _provider().complete([user("q")], temperature=0.1)

    def test_empty_choices_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        with pytest.raises(ProviderError, match="empty completion"):
            _provider().complete([user("q")], temperature=0.1)

    def test_null_content_raises_provider_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(f"{BASE_URL}/chat/completions").mock(return_value=_ok_response(None))
        with pytest.raises(ProviderError, match="empty completion"):
            _provider().complete([user("q")], temperature=0.1)
