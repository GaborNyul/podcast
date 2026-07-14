"""Tests for podcast.llm.anthropic (SDK client replaced by a typed double)."""

from dataclasses import dataclass, field
from typing import cast

import anthropic
import httpx
import pytest

from podcast.errors import ProviderError
from podcast.llm.anthropic import AnthropicProvider, prepare_schema
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
        expected: dict[str, object] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        assert call["output_config"] == {"format": {"type": "json_schema", "schema": expected}}

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


class TestPrepareSchema:
    """Pydantic schemas must be rewritten into the shape structured outputs accept."""

    def test_objects_get_additional_properties_false_recursively(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"turn": {"type": "object", "properties": {}}},
            "$defs": {"Turn": {"type": "object", "properties": {}}},
        }
        prepared = prepare_schema(schema)
        assert prepared["additionalProperties"] is False
        properties = cast("dict[str, dict[str, object]]", prepared["properties"])
        assert properties["turn"]["additionalProperties"] is False
        defs = cast("dict[str, dict[str, object]]", prepared["$defs"])
        assert defs["Turn"]["additionalProperties"] is False

    def test_unsupported_constraint_keywords_are_stripped(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "segments": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "minutes": {"type": "integer", "minimum": 1, "maximum": 30},
                "title": {"type": "string", "minLength": 1, "maxLength": 80},
            },
        }
        properties = cast("dict[str, dict[str, object]]", prepare_schema(schema)["properties"])
        assert properties["segments"] == {"type": "array", "items": {"type": "string"}}
        assert properties["minutes"] == {"type": "integer"}
        assert properties["title"] == {"type": "string"}

    def test_property_named_like_a_keyword_survives(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {"minimum": {"type": "string"}},
        }
        properties = cast("dict[str, dict[str, object]]", prepare_schema(schema)["properties"])
        assert properties["minimum"] == {"type": "string"}

    def test_data_carrying_keys_are_not_walked(self) -> None:
        schema: dict[str, object] = {
            "type": "object",
            "properties": {
                "angles": {"type": "object", "default": {"minimum": "keep"}},
                "kind": {"type": "string", "enum": ["minItems", "other"]},
            },
        }
        properties = cast("dict[str, dict[str, object]]", prepare_schema(schema)["properties"])
        assert properties["angles"]["default"] == {"minimum": "keep"}
        assert properties["kind"]["enum"] == ["minItems", "other"]

    def test_schema_valued_additional_properties_is_preserved(self) -> None:
        schema: dict[str, object] = {"type": "object", "additionalProperties": {"type": "string"}}
        assert prepare_schema(schema)["additionalProperties"] == {"type": "string"}

    def test_original_schema_is_not_mutated(self) -> None:
        schema: dict[str, object] = {"type": "object", "properties": {}, "minProperties": 1}
        prepare_schema(schema)
        assert schema == {"type": "object", "properties": {}, "minProperties": 1}

    def test_non_dict_nodes_pass_through(self) -> None:
        schema: dict[str, object] = {
            "anyOf": [{"type": "object", "properties": {}}, {"type": "null"}]
        }
        any_of = cast("list[dict[str, object]]", prepare_schema(schema)["anyOf"])
        assert any_of[0]["additionalProperties"] is False
        assert any_of[1] == {"type": "null"}


class TestCompleteErrors:
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
