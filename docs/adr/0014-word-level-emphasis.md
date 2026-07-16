# ADR 0014: Word-level emphasis markup (`*word*`) with per-engine rendering

Date: 2026-07-15
Status: Accepted

## Context

Line-level delivery notes (ADR 0010) set the register for a whole utterance, but a human
host leans on individual words; that stress is what makes a reveal land. A capability survey
(verified against engine source code, model cards, and maintainer statements) found no
cross-engine markup standard: none of qwen3, SoulX, kokoro-onnx, VibeVoice, or Chatterbox
supports SSML, and a published benchmark (arXiv 2508.17494) shows LLMs systematically
mangle SSML anyway. What does exist: markdown asterisks are native LLM output; CAPS on the
stressed word is the closest de-facto render-side convention (ElevenLabs guidance, Bark,
Chatterbox); qwen3's `instruct` channel accepts a clause naming the word (probabilistic);
and SoulX's released tokenizer ships undocumented `<|stress_start|>`/`<|stress_end|>`
single-ID added tokens — believed at decision time to be worst-case a prosodic no-op,
never spoken text (the hardware audition later disproved this; see Consequences). Raw
unknown markup reaching SoulX *is* spoken aloud, so markup must never pass through
unrendered.

## Decision

- The script layer marks stress inline in spoken text: `*word*` — single asterisks, a
  non-empty span with no `*` inside and no leading/trailing whitespace inside. The shared
  FORMAT FOR AUDIO prompt block carves this out as the single exception to the no-markdown
  rule and directs the LLM to use it sparingly (a word or two, only where a host would lean
  on it). Markup lives inside `Turn.text`: no model or artifact-schema change, and the
  span attaches to its word so `word_count()` and the `**Host:**` line grammar are
  unaffected.
- Trust boundaries differ: LLM output is normalized tolerantly (`emphasis.normalize`
  drops malformed or stray asterisks in `_dialogue_request`, covering generate, polish, and
  length-repair); hand-edited script.md is validated strictly (malformed emphasis raises
  `ScriptError` with the line number, like every other grammar violation).
- The polish pass may keep, sharpen, sparingly add, or move emphasis marks — the same
  mandate it has for delivery notes; the length-repair prompt keeps marks through rewrites.
- `EngineInfo` gains `supports_emphasis`; the CLI strips markup before caching and
  synthesis for engines that declare `False` (mirroring the ADR 0010 delivery pattern), so
  emphasis edits are cache-free on kokoro and re-render exactly the affected lines on
  supporting engines. Engines only ever see markup they declared support for.
- qwen3 renders a span as CAPS in the text plus an appended instruct clause naming the
  word ("Put strong emphasis on the word 'X'.") — best-effort by design (~35–50%
  correct-word hit rate in published fine-grained instruct benchmarks).
- SoulX renders a span as `<|stress_start|>span<|stress_end|>`, gated by the layered
  config flag `tts.soulx_stress_markup` (default off — see Consequences). The flag feeds
  `supports_emphasis`, so turning it off routes through the CLI strip path — cache
  invalidation stays correct with no new key component.
- Pacing heuristics (ADR 0011) read markup-stripped text so a trailing `*` never masks
  the interruption/backchannel/question detection.

## Consequences

- Scripts gain a hand-editable stress channel that degrades gracefully: unsupported
  engines get clean text, supporting engines get their native best effort.
- The deep-dive system prompt bytes changed: the SHA-256 pin and shared-block assertions
  in `tests/test_formats.py` were updated deliberately; all four formats inherit the rule
  via the shared `AUDIO_BLOCK`.
- Cache keys did not gain a field; the cache keys on the engine-visible text. On a
  supporting engine the marked text itself is engine-visible, so any emphasis edit
  re-renders that line — even one whose final engine payload is unchanged (e.g. adding an
  untreated span like `*100*`). Conservative over-invalidation: never stale audio,
  occasionally a redundant render. On non-supporting engines markup is stripped before
  keying, so emphasis edits stay cache-free.
- qwen3 emphasis is probabilistic, not guaranteed — the A/B listening check (plain vs CAPS
  vs instruct vs both) ran on hardware 2026-07-15 in two rounds and confirmed
  CAPS+instruct as the default. It also drove a per-span guard: treatment applies exactly
  when uppercasing changes a span of ≥3 chars; spans CAPS cannot change (all-caps,
  numerals) and short spans (CAPS read `*it*` as the acronym "eye-tee") get no treatment
  — the round-2 audition heard clause-alone stress the wrong words or nothing.
- SoulX's stress tokens proved not inert: the same audition heard them vocalize as
  garbage syllables (they encode as clean single IDs, so the embeddings are untrained
  upstream), so `tts.soulx_stress_markup` now defaults to off — the "flag turns the
  feature off without a code change" escape hatch became the default posture, and the
  flag is opt-in for experimentation.
- Future engines (VibeVoice, Chatterbox) slot in by declaring their capability and, if
  supported, their own renderer (Chatterbox: CAPS; VibeVoice: strip).
- The grammar reserves `*` outright: spoken text cannot carry a literal asterisk — the
  parser rejects hand-edited math like `3*4` with an instructive error, and the LLM
  boundary silently drops it. Two same-line asterisks may also legally pair into an
  arbitrarily long multi-word span (`3*4=12 and 4*3=12` parses as one span covering
  `4=12 and 4`); the sparing-use prompt guidance is the guardrail, and a parse-time
  warning on implausibly long spans is a possible refinement if hand edits ever hit this.
