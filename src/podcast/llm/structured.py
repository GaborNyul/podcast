# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Schema-guided completion: native enforcement when supported, prompt+retry fallback."""

import json
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ValidationError

from podcast.errors import ProviderError
from podcast.llm.base import ChatMessage, ChatProvider


def extract_json(text: str) -> str:
    """Strip code fences and surrounding prose from the outermost JSON document."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        closing = stripped.rfind("```")
        if first_newline != -1 and closing > first_newline:
            stripped = stripped[first_newline + 1 : closing].strip()
    candidates = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not candidates:
        raise ValueError("no JSON object or array found in model output")
    start = min(candidates)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise ValueError("unterminated JSON in model output")
    return stripped[start : end + 1]


def complete_structured[T: BaseModel](
    provider: ChatProvider,
    messages: Sequence[ChatMessage],
    output_type: type[T],
    *,
    temperature: float,
    max_retries: int = 2,
    schema: Mapping[str, object] | None = None,
) -> T:
    """Get a validated `output_type` from the provider, retrying with error feedback.

    `schema` overrides the model-derived JSON schema — used to inject dynamic
    constraints (e.g. a speaker-name enum) while validating with a static model.
    """
    if schema is None:
        schema = output_type.model_json_schema()
    instruction = ChatMessage(
        role="user",
        content=(
            "Respond with ONLY a JSON document matching this JSON schema — "
            f"no prose, no code fences:\n{json.dumps(schema)}"
        ),
    )
    attempt_messages: list[ChatMessage] = [*messages, instruction]
    last_error = ""
    for _attempt in range(max_retries + 1):
        text = provider.complete(attempt_messages, temperature=temperature, json_schema=schema)
        try:
            return output_type.model_validate_json(extract_json(text))
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            attempt_messages = [
                *messages,
                instruction,
                ChatMessage(role="assistant", content=text),
                ChatMessage(
                    role="user",
                    content=(
                        f"That reply failed validation:\n{last_error}\n"
                        "Respond again with ONLY corrected JSON matching the schema."
                    ),
                ),
            ]
    raise ProviderError(
        f"structured output from {provider.name} failed after "
        f"{max_retries + 1} attempts: {last_error}"
    )
