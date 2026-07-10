"""Claude provider via the official Anthropic SDK."""

from collections.abc import Mapping, Sequence

import anthropic

from podcast.errors import ProviderError
from podcast.llm.base import ChatMessage

_MAX_TOKENS = 16000


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
                    {"format": {"type": "json_schema", "schema": dict(json_schema)}}
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
