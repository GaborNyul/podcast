"""Tests for podcast.llm.structured."""

from collections.abc import Mapping, Sequence

import pytest
from pydantic import BaseModel

from podcast.errors import ProviderError
from podcast.llm.base import ChatMessage, user
from podcast.llm.structured import complete_structured, extract_json


class Answer(BaseModel):
    value: int
    label: str


class _ScriptedProvider:
    """Returns pre-planned replies; records every call it receives."""

    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        del temperature, json_schema
        self.calls.append(list(messages))
        return self.replies[len(self.calls) - 1]


class TestExtractJson:
    def test_bare_object(self) -> None:
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_bare_array(self) -> None:
        assert extract_json("[1, 2]") == "[1, 2]"

    def test_fenced_json(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fence_without_language_tag(self) -> None:
        assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_unclosed_fence_still_extracts(self) -> None:
        assert extract_json('```{"a": 1}```') == '{"a": 1}'

    def test_prose_around_json(self) -> None:
        assert extract_json('Sure!\n{"a": 1}\nHope that helps.') == '{"a": 1}'

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="no JSON"):
            extract_json("I could not produce anything structured.")

    def test_unterminated_json_raises(self) -> None:
        with pytest.raises(ValueError, match="unterminated"):
            extract_json('{"a": 1')

    def test_unicode_content_survives(self) -> None:
        assert extract_json('{"a": "műsor 🎙️"}') == '{"a": "műsor 🎙️"}'


class TestCompleteStructured:
    def test_valid_first_reply(self) -> None:
        provider = _ScriptedProvider(['{"value": 3, "label": "ok"}'])
        result = complete_structured(provider, [user("count")], Answer, temperature=0.2)
        assert result == Answer(value=3, label="ok")
        assert len(provider.calls) == 1
        assert "JSON schema" in provider.calls[0][-1].content

    def test_retry_feeds_back_validation_error(self) -> None:
        provider = _ScriptedProvider(
            ["not json at all", '{"value": "x"}', '{"value": 7, "label": "fixed"}']
        )
        result = complete_structured(
            provider, [user("count")], Answer, temperature=0.2, max_retries=2
        )
        assert result.value == 7
        assert len(provider.calls) == 3
        assert "failed validation" in provider.calls[1][-1].content
        assert provider.calls[1][-2].role == "assistant"

    def test_exhausted_retries_raise_provider_error(self) -> None:
        provider = _ScriptedProvider(["nope", "still nope"])
        with pytest.raises(ProviderError, match="after 2 attempts"):
            complete_structured(provider, [user("count")], Answer, temperature=0.2, max_retries=1)

    def test_fenced_reply_is_accepted(self) -> None:
        provider = _ScriptedProvider(['```json\n{"value": 1, "label": "f"}\n```'])
        result = complete_structured(provider, [user("go")], Answer, temperature=0.0)
        assert result.label == "f"
