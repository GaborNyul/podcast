"""Tests for podcast.llm.anthropic (SDK client replaced by a typed double)."""

from dataclasses import dataclass, field

import anthropic
import httpx
import pytest

from podcast.errors import ProviderError
from podcast.llm.anthropic import AnthropicProvider
from podcast.llm.base import system, user


@dataclass(frozen=True)
class _TextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class _ThinkingBlock:
    thinking: str = ""
    type: str = "thinking"


@dataclass(frozen=True)
class _Response:
    content: list[object]
    stop_reason: str = "end_turn"


def _empty_calls() -> list[dict[str, object]]:
    return []


@dataclass
class _Messages:
    response: _Response
    error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=_empty_calls)

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    last: "_FakeClient | None" = None

    def __init__(self, *, api_key: str | None = None, timeout: float = 600.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.messages = _Messages(_Response([_TextBlock("fine answer")]))
        _FakeClient.last = self


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    monkeypatch.setattr("podcast.llm.anthropic.anthropic.Anthropic", _FakeClient)
    return _FakeClient


def _last_client() -> _FakeClient:
    assert _FakeClient.last is not None
    return _FakeClient.last


class TestComplete:
    @pytest.mark.usefixtures("fake_client")
    def test_returns_joined_text_blocks(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        client = _last_client()
        client.messages.response = _Response(
            [_ThinkingBlock(), _TextBlock("part one "), _TextBlock("part two")]
        )
        assert provider.complete([user("q")], temperature=0.5) == "part one part two"

    @pytest.mark.usefixtures("fake_client")
    def test_system_messages_join_into_system_param(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        provider.complete([system("rule one"), system("rule two"), user("q")], temperature=0.5)
        call = _last_client().messages.calls[-1]
        assert call["system"] == "rule one\n\nrule two"
        assert call["messages"] == [{"role": "user", "content": "q"}]
        assert call["thinking"] == {"type": "adaptive"}

    @pytest.mark.usefixtures("fake_client")
    def test_temperature_is_not_forwarded(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        provider.complete([user("q")], temperature=0.9)
        assert "temperature" not in _last_client().messages.calls[-1]

    @pytest.mark.usefixtures("fake_client")
    def test_schema_becomes_output_config(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        schema: dict[str, object] = {"type": "object", "properties": {}}
        provider.complete([user("q")], temperature=0.5, json_schema=schema)
        call = _last_client().messages.calls[-1]
        assert call["output_config"] == {"format": {"type": "json_schema", "schema": schema}}

    @pytest.mark.usefixtures("fake_client")
    def test_no_schema_leaves_output_config_not_given(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        provider.complete([user("q")], temperature=0.5)
        assert _last_client().messages.calls[-1]["output_config"] is anthropic.omit

    @pytest.mark.usefixtures("fake_client")
    def test_refusal_raises_provider_error(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        client = _last_client()
        client.messages.response = _Response([], stop_reason="refusal")
        with pytest.raises(ProviderError, match="refusal"):
            provider.complete([user("q")], temperature=0.5)

    @pytest.mark.usefixtures("fake_client")
    def test_empty_text_raises_provider_error(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        client = _last_client()
        client.messages.response = _Response([_ThinkingBlock()])
        with pytest.raises(ProviderError, match="empty completion"):
            provider.complete([user("q")], temperature=0.5)

    @pytest.mark.usefixtures("fake_client")
    def test_status_error_maps_to_provider_error(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        client = _last_client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.messages.error = anthropic.APIStatusError(
            "overloaded",
            response=httpx.Response(529, request=request),
            body=None,
        )
        with pytest.raises(ProviderError, match="HTTP 529"):
            provider.complete([user("q")], temperature=0.5)

    @pytest.mark.usefixtures("fake_client")
    def test_connection_error_maps_to_provider_error(self) -> None:
        provider = AnthropicProvider(model="claude-opus-4-8")
        client = _last_client()
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        client.messages.error = anthropic.APIConnectionError(request=request)
        with pytest.raises(ProviderError, match="cannot reach anthropic"):
            provider.complete([user("q")], temperature=0.5)

    @pytest.mark.usefixtures("fake_client")
    def test_api_key_and_timeout_reach_the_client(self) -> None:
        AnthropicProvider(
            model="claude-opus-4-8",
            api_key="sk-ant-test",  # pragma: allowlist secret
            timeout_seconds=42.0,
        )
        client = _last_client()
        assert client.api_key == "sk-ant-test"  # pragma: allowlist secret
        assert client.timeout == 42.0
