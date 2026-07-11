# ADR 0011: Expressiveness polish — sampling, pacing, and a disfluency rewrite pass

Date: 2026-07-11
Status: Accepted

## Context

ADR 0010 gave the pipeline a per-turn emotion channel. Three cheaper levers from the same
research remained: (1) Qwen3-TTS reads robotic at low sampling temperature (community
consensus: temperature 0.8, top_p 0.9, repetition_penalty ~1.05); (2) pause structure is a
measured affect channel, but assembly sampled all inter-turn silences from one uniform
range; (3) NotebookLM's script pipeline ends with a dedicated disfluency-injection pass
("you cannot listen to two robots talking"), while ours wrote dialogue in one shot.

## Decision

- `TTSSettings` exposes `qwen3_temperature/qwen3_top_p/qwen3_repetition_penalty`
  (defaults 0.8/0.9/1.05); the qwen3 engine forwards them on every generate call.
- `podcast.audio.pacing` derives one scale per inter-turn gap from structural cues:
  interruptions (trailing dash) 0.4, backchannels (≤4-word reply) 0.5, question hand-offs
  0.7, ellipses 1.4, else 1.0. `assemble_episode` accepts `gap_scales` and multiplies the
  sampled pause; silences still round to the 50 ms reuse grid.
- `HostSpec.style` is a per-host baseline performance direction composed with each line's
  delivery note (`"style; note"`) before it reaches a delivery-capable engine. Measured
  motivation: the vivian voice reads English at ~116 wpm while ryan reads ~159; the instruct
  "Speak at a fast, energetic pace." lifts vivian to ~148 wpm. The composed string joins the
  cache key, so changing a host's style re-renders that host's lines.
- `polish_dialogue` (gated by `script.polish_pass`, default on) rewrites the whole draft
  once for radio texture — disfluencies, interruptions, breaking up stacked long turns,
  sharpening delivery notes — with facts, attributions, and rituals pinned by `POLISH_BRIEF`
  and the word target restated ("approximately N words", the FakeProvider contract phrase).
  Order: write_dialogue → polish_dialogue → ensure_length, so length drift from the rewrite
  is repaired by the existing pass.

## Consequences

- Generation costs one extra whole-script LLM call when polish is on; `polish_pass = false`
  restores the old behavior.
- Pacing needs no configuration and applies to every engine, since it acts at assembly time.
- The pacing heuristics read text, not delivery notes — they work on pre-ADR-0010 scripts
  and hand edits alike.
