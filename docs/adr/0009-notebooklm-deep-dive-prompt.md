# ADR 0009: NotebookLM-style "Deep Dive" system prompt (hybrid-v2)

## Status

Accepted (2026-07-11)

## Context

The original `SYSTEM_PROMPT` in `podcast.script.pipeline` was four sentences of generic
guidance. In a six-way judged bake-off of NotebookLM-Audio-Overview-style prompts (all
generating from the same source document under identical harness rules, scored by four
anonymized judge lenses: style fidelity, naturalness by ear, faithfulness audit, listener
preference), it placed **last** on style — judges unanimously described its output as "an
audio textbook": interviewer-lecturer alternation, list-recital monologues, no listener
address. It did win the faithfulness audit, so grounding language had to be preserved.

The bake-off candidates were synthesized from web research into how NotebookLM's Audio
Overviews are actually prompted and why they sound natural: Nicole Hennig's transcript-based
style reconstruction, the Jaden Geller on-air prompt leak (via Simon Willison) and related
Reddit probes, a Latent.Space interview with the NotebookLM team (multi-pass pipeline,
deliberate withholding, "too much agreement isn't fun to listen to"), and the prompts of
open-notebooklm, podcastfy, Meta NotebookLlama, and Mozilla document-to-podcast.

The winner ("hybrid", a best-of synthesis) took 3 of 4 lenses, was then refined against the
judges' criticisms (headline-topic coverage, attribution hygiene, producer-speak ban,
affirmation rotation, one-named-concept-per-turn, earned persuasion, stake-bearing
questions, canonical outro) and the refined **hybrid-v2** beat the original 3-0 in a
counterbalanced head-to-head, including a clear verdict on faithfulness/coverage.

## Decision

Adopt hybrid-v2 as the generation prompt, split to fit the outline -> per-segment pipeline
in a new `podcast.script.prompts` module:

- `SYSTEM_PROMPT` — the Deep Dive engine: listener persona, guide/companion host
  archetypes (mapped onto the configured hosts via their personas), conversation engine,
  grounding + attribution hygiene, a compact arc description, and the TTS format rules the
  bake-off harness had supplied externally (no bracketed cues — `[laughs]` is unspeakable).
- `OUTLINE_BRIEF` — the headline-coverage rule runs at planning time, where segment
  selection actually happens.
- `OPENING_POSITION` / `CONTINUING_POSITION` / `FINAL_POSITION` — the cold-open hook and
  the outro ritual ride the per-segment position hints, so mid-episode segments are never
  tempted to re-open the show or sign off early.

Default host personas in `podcast.config` now name the roles ("the companion", "the
guide") so role assignment is explicit rather than inferred.

## Consequences

- Generated scripts follow the NotebookLM house style: ritual open/close, backchannel glue,
  analogies with callbacks, sustained-skepticism exchanges, listener-directed "you".
- The prompt is written for a two-host dynamic; config allows more hosts, and the prompt
  explicitly assigns any extra hosts to share the companion role.
- Word-budget control, speaker-enum schema injection, and the `approximately N words`
  phrasing (which the deterministic FakeProvider parses) are unchanged.
- Evaluation artifacts (all six candidate prompts/scripts, rubric, judge verdicts, rendered
  episode) live in `episodes/rag-deep-dive/` locally; that directory is gitignored, so this
  ADR records the provenance essentials.
