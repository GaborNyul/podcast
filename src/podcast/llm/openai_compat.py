"""One OpenAI-compatible chat transport serving ollama, ollama-cloud, openai, copilot."""

from collections.abc import Mapping, Sequence

import httpx
from pydantic import BaseModel, ValidationError

from podcast.errors import ProviderError
from podcast.llm.base import ChatMessage


class _ChoiceMessage(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _ChoiceMessage


class _ChatResponse(BaseModel):
    choices: list[_Choice] = []


class OpenAICompatProvider:
    """Chat-completions transport parameterized by base_url/auth/quirks."""

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        extra_headers: Mapping[str, str] | None = None,
        supports_json_schema: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self._supports_json_schema = supports_json_schema
        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers is not None:
            headers.update(extra_headers)
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_seconds)

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        json_schema: Mapping[str, object] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "stream": False,
        }
        if json_schema is not None and self._supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": dict(json_schema), "strict": True},
            }
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            snippet = exc.response.text[:500]
            raise ProviderError(
                f"{self.name} returned HTTP {exc.response.status_code}: {snippet}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot reach {self.name}: {exc}") from exc
        try:
            parsed = _ChatResponse.model_validate_json(response.text)
        except ValidationError as exc:
            raise ProviderError(f"malformed response from {self.name}: {exc}") from exc
        if not parsed.choices or parsed.choices[0].message.content is None:
            raise ProviderError(f"empty completion from {self.name}")
        return parsed.choices[0].message.content
