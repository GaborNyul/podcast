# ADR 0004: Qwen3-TTS on TheRock wheels behind an extra; Kokoro CPU fallback in core

Date: 2026-07-10
Status: Accepted

## Context

gfx1151 (Strix Halo) needs AMD "TheRock" nightly torch wheels
(`https://rocm.nightlies.amd.com/v2/gfx1151/`); stock ROCm wheels fail, and
`HSA_OVERRIDE_GFX_VERSION=11.0.0` must not be set. Conv-heavy models can run slower
on this GPU than on CPU, so device choice must be per-engine. A torch install is
heavy and machine-specific.

## Decision

- Core install is torch-free; Kokoro-82M via `kokoro-onnx` (CPU, ONNX) is the
  works-everywhere default fallback.
- Qwen3-TTS-12Hz-1.7B (GPU default on the target box) lives behind the
  `podcast[qwen3]` extra with torch routed to the TheRock gfx1151 index via uv
  per-package index pinning, pinned to a known-good nightly.
- `qwen3.py` lazy-imports torch; engines declare their preferred device; `doctor`
  verifies GPU visibility.

## Consequences

- The tool works on any machine out of the box; GPU quality is opt-in.
- TheRock nightly churn is contained to one pinned index reference and one adapter
  module; hardware verification happens on the target box (RTF benchmark,
  integration-marked).
