# ADR 0002: Two-stage resumable workflow with an editable script contract

Date: 2026-07-10
Status: Accepted

## Context

Script generation (LLM) and audio synthesis (TTS) are slow, independently fallible,
and the user wants to review/edit the dialogue before spending synthesis time.

## Decision

Two explicit stages sharing an on-disk episode workspace
(`episodes/<slug>/`): `generate` produces `script.md` (+ `transcript.json` sidecar,
`sources.json`, `outline.json`); `synthesize` consumes it; `create` chains both.
`script.md` ⇄ `Transcript` round-trips losslessly (front-matter + `**Host:**` turn
lines), enforced by a hypothesis property test — hand edits always survive re-parsing.

## Consequences

- Any stage can be re-run in isolation; the workspace is the only state.
- The markdown grammar is a public contract; parser changes require the round-trip
  gate to stay green.
