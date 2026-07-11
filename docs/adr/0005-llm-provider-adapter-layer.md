# ADR 0005: LLM provider adapter layer with a shared OpenAI-compatible transport

Date: 2026-07-10
Status: Accepted

## Context

Script generation must run against local Ollama by default but also Ollama Cloud,
OpenAI, Claude, and GitHub Copilot, and be extensible. Four of the five speak the
OpenAI chat-completions dialect; tests and offline e2e need no network at all.

## Decision

A `ChatProvider` protocol with a registry (name → factory). One
`openai_compat.py` transport implements ollama/ollama-cloud/openai/copilot as
configuration (base_url + auth + quirks); Claude uses the native anthropic SDK;
`fake.py` is a deterministic canned provider for tests and offline e2e. Structured
output goes through one helper: native json_schema where supported, else
schema-in-prompt + fence-strip + pydantic validation with bounded retry.

## Consequences

- Adding a provider is a registry entry (plus auth quirks at most).
- Copilot's device-flow/token-exchange complexity is isolated in `copilot_auth.py`;
  GitHub Models API (PAT, OpenAI-compatible) is the documented fallback if the
  endpoint proves unstable.
