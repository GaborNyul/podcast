# ADR 0007: Content-addressed per-line segment cache

Date: 2026-07-10
Status: Accepted

## Context

Synthesis is the slowest stage. The editable-script workflow means users tweak a few
lines and re-synthesize; re-rendering the whole episode would waste minutes per edit.

## Decision

Each turn's audio is cached as `segments/<sha256(engine, voice, text)>.wav` in the
episode workspace. `synthesize` renders only cache misses and reports hit/miss counts.

## Consequences

- Editing one line re-renders exactly one segment.
- Changing engine or voice mapping naturally invalidates affected segments (key
  includes both); stale segments are orphaned files, cleaned by re-assembly ignoring
  unreferenced hashes.
