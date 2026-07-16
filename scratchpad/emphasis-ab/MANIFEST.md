# Emphasis A/B listening demos (ADR 0014 — final hardware validation)

Rendered on the gfx1151/Strix Halo box by `render.py` via the public engine APIs
(`Qwen3Engine.synthesize_line`, `SoulXEngine.synthesize_dialogue`). All files in `out/`.

## Sentences

- **S1** `The bottleneck wasn't retrieval — it was the *reranker* all along.` — plain mid-sentence mark
- **S2** `*Not* every query needs retrieval.` — sentence-initial mark; CAPS render yields `NOT` (mixed-case span, checks case matching between CAPS text and instruct clause)
- **S3** `That's *it* — the entire fix was one line.` — short-word span; CAPS yields `IT` (acronym misread risk: "eye-tee")
- **S4** `Plain *RAG* retrieves neighbors of the question, not the answer.` — already-uppercase span; CAPS render is a no-op, the instruct clause alone must do the work
- **S5** (SoulX only) `And that is why hybrid search wins. Wait — you are telling me the fix was *free*` — emphasized unpunctuated FINAL word (checks upstream auto-period interaction)

## qwen3 arms (voice `vivian`, no host style)

| arm | text handed to engine | instruct channel |
|---|---|---|
| `plain` | markup stripped | (empty) |
| `caps` | pre-rendered CAPS — engine sees no spans, adds no clause | (empty) |
| `instruct` | markup stripped | `Put strong emphasis on the word '<span>'.` |
| `both` | marked text as-is — the SHIPPED path (engine renders CAPS + appends clause itself) | (engine-built clause) |

## Files — qwen3 (16)

| file | listen FOR |
|---|---|
| `qwen3-S1-plain.wav` | Baseline: where does natural stress fall without any nudging? |
| `qwen3-S1-caps.wav` | Does stress land on "reranker"? Any artifact from CAPS (shouting, spelled-out read)? |
| `qwen3-S1-instruct.wav` | Does the clause alone move stress to "reranker"? (~35–50% hit rate expected) |
| `qwen3-S1-both.wav` | Shipped path: stress on "reranker" more reliable than either single arm? |
| `qwen3-S2-plain.wav` | Baseline for sentence-initial stress. |
| `qwen3-S2-caps.wav` | Does "NOT" get stress, or get read as an acronym/shout? |
| `qwen3-S2-instruct.wav` | Clause names lowercase-marked 'Not' — does case mismatch confuse it? |
| `qwen3-S2-both.wav` | Text says NOT, clause says 'Not' — do the two channels still agree? |
| `qwen3-S3-plain.wav` | Baseline; "it" is normally unstressed. |
| `qwen3-S3-caps.wav` | CRITICAL: is "IT" spelled out as "eye-tee"? |
| `qwen3-S3-instruct.wav` | Can the clause stress a function word without the CAPS misread risk? |
| `qwen3-S3-both.wav` | Shipped path on the riskiest span: stress vs "eye-tee" misread. |
| `qwen3-S4-plain.wav` | Baseline: is "RAG" already read as a word (not spelled out)? |
| `qwen3-S4-caps.wav` | CAPS is a NO-OP here — should sound identical to plain (any diff is sampling noise). |
| `qwen3-S4-instruct.wav` | The clause alone must do the work: does "RAG" gain stress vs plain? |
| `qwen3-S4-both.wav` | Same as instruct in effect (CAPS no-op) — does the appended clause still land it? |

## Files — SoulX (6)

3-line mini-dialogue, `alex`/`maya` repo refs; line 2 = S1 (Maya), line 3 = S5 (Alex, final line).
`marked` = engine converts `*span*` to `<|stress_start|>span<|stress_end|>` tokens; `stripped` = same words, no tokens (control).

| file | line | listen FOR |
|---|---|---|
| `soulx-marked-1.wav` | Alex: "So we spent the whole sprint blaming the vector index." | Unmarked line: should match stripped-1 in register (dialogue context is identical). |
| `soulx-marked-2.wav` | Maya: S1 | Does stress land on "reranker"? Any glitch/audible artifact at token positions? |
| `soulx-marked-3.wav` | Alex: S5 | Does the FINAL word "free" get stress? Does the sentence still terminate cleanly (no cutoff/run-on) with the mark on the last unpunctuated word? |
| `soulx-stripped-1.wav` | Alex line 1 | Control twin of marked-1. |
| `soulx-stripped-2.wav` | Maya: S1 stripped | Control: marked vs stripped — any audible difference AT ALL? (Tokens may be inert — that's a finding, not a failure.) |
| `soulx-stripped-3.wav` | Alex: S5 stripped | Control: clean termination without tokens, for comparison with marked-3. |

## Notes

- One model load per engine, reused across all of its renders (run per-engine: `render.py qwen3`, then `render.py soulx`).
- qwen3 sampling is stochastic (temperature 0.8): differences between arms need to be bigger than run-to-run noise before crediting the markup.

## Results (user audition, 2026-07-15 — verdicts encoded in commits 30266f1 + d449e98)

- **qwen3**: `both` (CAPS + clause) won S1 and S2 outright; clause-alone landed the target
  word 0-for-5 across both rounds (it *calibrates* CAPS, it does not replace it).
- **S3 confirmed the acronym misread**: CAPS "IT" was spoken "eye-tee" in both CAPS arms.
- **S4/S6 round 2**: clause-alone on all-caps spans stressed the wrong word or nothing.
- **Shipped rule** (`qwen3.py _treated`): a span is CAPSed + clause-named exactly when
  uppercasing changes it and it is ≥3 chars; all-caps spans, numerals, and ≤2-char spans
  get no treatment.
- **SoulX stress tokens are NOT inert — they vocalize as garbage syllables** at the token
  positions (marked-2 "ranker" + junk, marked-3 "ickt" before "free"), despite encoding as
  clean single IDs (151712/151713) — untrained upstream embeddings. `soulx_stress_markup`
  therefore defaults to **off**.
- Sampling-noise calibration: S4 `plain` vs `caps` were byte-identical inputs yet judged
  differently — single-clip differences below that bar prove nothing.
- `render_v2.py` re-rendered S3/S4 plus a new S6 (`Turns out *AI* is doing the reranking
  here.`) through the guarded engine (`qwen3-S{3,4,6}-both-v2.wav`), confirming no more
  "eye-tee" and driving the round-2 all-caps flip.
