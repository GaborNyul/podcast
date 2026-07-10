# ADR 0001: Python 3.13 managed by uv

Date: 2026-07-10
Status: Accepted

## Context

Standards v3 prefers Python 3.14+ but explicitly allows 3.13 for projects with
dependency constraints. The TTS stack (torch TheRock nightlies for gfx1151,
onnxruntime, kokoro-onnx) does not yet ship 3.14 wheels.

## Decision

Target Python 3.13 exclusively (`requires-python = ">=3.13,<3.14"`), managed by uv
with a committed hash-pinned `uv.lock`. No t-strings (PEP 750 is 3.14+); all other
standards v3 language rules apply unchanged (no `Any`, PEP 604 unions, PEP 695 type
aliases).

## Consequences

- Type-checker and ruff configs pin 3.13 where the standards examples show 3.14.
- Revisit when the torch/ONNX ecosystem ships 3.14 wheels; migration is a version bump.
