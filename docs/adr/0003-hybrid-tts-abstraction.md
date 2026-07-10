# ADR 0003: Hybrid TTS abstraction — stitched single-speaker now, dialogue-native later

Date: 2026-07-10
Status: Accepted

## Context

The best verified local engines (Qwen3-TTS, Kokoro) are single-speaker per call.
Dialogue-native models (VibeVoice, MOSS-TTSD) promise better turn-taking but have
license/provenance/perf caveats on this hardware (July 2026).

## Decision

`TTSEngine` protocol exposes `capabilities()` (`dialogue_native`, preferred device)
plus `synthesize_line()`; podcasts are rendered line-by-line and stitched. A
`synthesize_dialogue()` slot exists for future dialogue-native engines so they can
render whole conversations in one pass without pipeline changes.

## Consequences

- MVP ships with stitched engines only; naturalness of turn-taking is bounded by
  per-line synthesis + inter-turn silence heuristics.
- Adding MOSS-TTSD later is an engine registration, not a pipeline rework.
