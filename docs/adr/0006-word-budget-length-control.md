# ADR 0006: Word-budget outlining as the only length control

Date: 2026-07-10
Status: Accepted

## Context

Prior art (SurfSense, podcastfy, open-notebooklm) shows LLMs cannot hit a target
duration from a prompt alone; only explicit per-segment word budgets work.

## Decision

`minutes → words` via `150 wpm × minutes × calibration` (default 0.85; measured
per-engine in the phase-6 benchmark and stored in config). The outline stage
allocates `target_words` per segment summing to the budget; dialogue generation is
per-segment; a repair pass expands/compresses any segment more than 15 % off target.

## Consequences

- Episode duration is controllable within ±15 % once the engine calibration is
  measured.
- The calibration factor is per-engine config, not a constant — rendered-audio wpm
  differs between Qwen3 and Kokoro.
