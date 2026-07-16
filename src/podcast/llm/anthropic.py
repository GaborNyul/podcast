# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Claude provider via the official Anthropic SDK."""

from collections.abc import Mapping, Sequence
from typing import cast

import anthropic

from podcast.errors import ProviderError
from podcast.llm.base import ChatMessage

_MAX_TOKENS = 16000

# JSON-schema keywords Anthropic structured outputs reject; complete_structured
# re-validates the parsed reply with pydantic, so dropping them loses nothing.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)
# Structural keys whose values map names to sub-schemas (walk values, keep keys).
_SCHEMA_MAPS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
# Keys carrying data, not schema — never walked, never stripped.
_DATA_KEYS = frozenset({"default", "examples", "const", "enum"})


def prepare_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """Rewrite a pydantic JSON schema into the shape structured outputs accept.

    Every object schema gets `additionalProperties: false` (required by the API)
    and unsupported constraint keywords are stripped. A schema-valued
    additionalProperties (dict fields) is kept as-is: forcing it to false would
    silently forbid every key, so the API's own error is the better failure.
    """

    def walk(node: object) -> object:
        if isinstance(node, list):
            return [walk(item) for item in cast("list[object]", node)]
        if not isinstance(node, Mapping):
            return node
        prepared: dict[str, object] = {}
        for key, value in cast("Mapping[str, object]", node).items():
            if key in _UNSUPPORTED_KEYWORDS:
                continue
            if key in _DATA_KEYS:
                prepared[key] = value
            elif key in _SCHEMA_MAPS and isinstance(value, Mapping):
                sub_schemas = cast("Mapping[str, object]", value)
                prepared[key] = {name: walk(sub) for name, sub in sub_schemas.items()}
            else:
                prepared[key] = walk(value)
        is_object = prepared.get("type") == "object" or "properties" in prepared
        if is_object and not isinstance(prepared.get("additionalProperties"), Mapping):
            prepared["additionalProperties"] = False
        return prepared

    return cast("dict[str, object]", walk(schema))


class AnthropicProvider:
    """Claude chat provider; structured output via output_config json_schema."""

    name = "anthropic"

    def __init__(
        self, *, model: str, api_key: str | None = None, timeout_seconds: float = 300.0
    ) -> None:
        self.model = model
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant` profile.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        # Current Claude models reject sampling parameters (400); steering happens
        # through the prompt, so the requested temperature is intentionally unused.
        del temperature
        system_parts = [message.content for message in messages if message.role == "system"]
        chat: list[anthropic.types.MessageParam] = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != "system"
        ]
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                thinking={"type": "adaptive"},
                system="\n\n".join(system_parts) if system_parts else anthropic.omit,
                output_config=(
                    {"format": {"type": "json_schema", "schema": prepare_schema(json_schema)}}
                    if json_schema is not None
                    else anthropic.omit
                ),
                messages=chat,
            )
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"anthropic returned HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"cannot reach anthropic: {exc}") from exc
        if response.stop_reason == "refusal":
            raise ProviderError("anthropic declined the request (stop_reason=refusal)")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise ProviderError("empty completion from anthropic")
        return text
