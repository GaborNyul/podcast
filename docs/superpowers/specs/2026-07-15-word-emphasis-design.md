# Word-level emphasis within a sentence ("micro" emotions) — design

Date: 2026-07-15
Status: Approved (TODO item #4; implemented on `feat/word-emphasis`; decision record in
ADR 0014)

## Problem

Line-level delivery notes (ADR 0010) are too coarse: they set the register for a whole
utterance, but a human host leans on individual words ("it *never* touches the network").
Two halves to close: the script LLM must mark the words to stress, and each TTS engine must
render (or consciously drop) that stress.

## Research findings (11-agent recon, 2026-07-15)

Verified against engine source code, model cards, tokenizer files, and maintainer replies:

| Engine | Word-stress channel | Render strategy |
|---|---|---|
| qwen3 CustomVoice | none documented; text enters LM verbatim; `instruct` accepts free text | CAPS on span + instruct clause naming the word (probabilistic, ~35–50%) |
| SoulX-Podcast | 5 documented paralinguistic tags; tokenizer ships undocumented `<|stress_start|>`/`<|stress_end|>` added tokens; unknown markup is spoken aloud | wrap span in stress tokens, config-gated |
| kokoro (kokoro-onnx) | none — espeak-ng G2P path ignores misaki stress markup | strip |
| VibeVoice (future) | none, maintainer-confirmed | strip |
| Chatterbox (future) | CAPS ("capitalization shifts emphasis", vendor doc); exaggeration is per-utterance | CAPS |

Cross-engine: nobody supports SSML, and LLMs measurably mangle SSML when asked to emit it
(arXiv 2508.17494); markdown asterisks are native LLM output; CAPS is the closest de-facto
render convention (ElevenLabs, Bark). Conclusion: mark in markdown, render per engine.

## Decisions (user-approved)

1. **Markup**: `*word*` single asterisks in `Turn.text`. Non-empty span, no `*` inside, no
   leading/trailing whitespace inside; `**bold**` and stray asterisks are invalid. Spans
   attach to words, so `word_count()` and the `**Host:**` line grammar are unaffected; no
   model/schema change.
2. **SoulX**: try the undocumented stress tokens, behind layered config flag
   `tts.soulx_stress_markup` (default on); flag feeds `EngineInfo.supports_emphasis`, so
   off ⇒ CLI strips (correct cache behavior for free).
3. **Polish pass**: may preserve **and sharpen** — keep existing marks, sparingly add or
   move them where the register moves (same mandate as delivery notes).
4. **Validation**: A/B listening demos rendered from `~/Downloads/RAG.md` — qwen3 {plain,
   CAPS, instruct, CAPS+instruct}, SoulX {tokens, stripped} — auditioned before pinning
   the qwen3 default (CAPS+instruct until then).

## Architecture

```
LLM (FORMAT FOR AUDIO teaches *word*, shared AUDIO_BLOCK → all 4 formats)
  → emphasis.normalize()      tolerant: drop malformed asterisks   (_dialogue_request;
                              covers generate / polish / length-repair)
  → script.md                 markup rides in turn text; _EDIT_HINT documents it
  → emphasis.validate()       strict: ScriptError + line number    (markdown_to_transcript)
  → CLI _run_synthesize       supports_emphasis? keep : strip_markup()   (mirrors ADR 0010
                              delivery blanking; applies to per-line + dialogue paths)
  → engine renderer           qwen3: render_caps + instruct clause
                              soulx: <|stress_start|>span<|stress_end|> in tagged_text
  → segment cache             keys on the text handed over ⇒ emphasis edits re-render
                              only where they audibly matter
```

New module `src/podcast/emphasis.py` (stdlib-only, importable from script and tts layers):
`EMPHASIS_RE`, `spans()`, `strip_markup()`, `normalize()`, `validate()`, `render_caps()`.
Pacing heuristics read `strip_markup(turn.text)` so a trailing `*` never masks
interruption/backchannel/question detection.

## Error handling

- LLM boundary: never fail the pipeline on bad marks — `normalize()` silently repairs.
- Hand-edit boundary: fail loud — `ScriptError` with line number, matching the parser's
  existing contract.
- Engine boundary: engines only receive markup they declared support for; unknown-tag
  leakage into SoulX (which speaks it aloud) is structurally impossible.

## Testing

TDD throughout; 100% coverage gate stays. New `tests/test_emphasis.py` (unit + Hypothesis:
normalize idempotent, validate∘normalize never raises); round-trip strategy in
`test_script_markdown.py` generates emphasis; deliberate SHA-256 re-pin of the deep-dive
prompt and shared-block asserts in `test_formats.py`; exact-payload pins in
`test_tts_qwen3.py` (CAPS text + instruct clause) and `test_tts_soulx.py` (stress tokens ×
flag states); cache-semantics tests in `test_app.py` (emphasis edit = cache hit on
non-supporting engine, re-render on supporting engine, dialogue digest includes rendered
text).

## Out of scope

Emphasis rendering for engines not yet in the repo (VibeVoice, Chatterbox — survey results
recorded above and in ADR 0014); misaki `[word](2)` stress (unreachable via kokoro-onnx);
multi-level emphasis (`**strong**`); STT-based empirical refinement of how often the LLM
should mark (TODO item #2 covers that loop).
