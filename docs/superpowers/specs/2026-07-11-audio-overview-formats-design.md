# Audio Overview Formats: Brief / Deep Dive / Debate / Critique

## Context

NotebookLM offers four Audio Overview formats — Brief (short fast summary), Deep Dive (long
detailed chat), Debate (hosts argue opposing viewpoints), Critique (analytical review of the
source). The generator currently hard-wires one format: the hybrid-v2 "Deep Dive" prompt
(ADR 0009) in `src/podcast/script/prompts.py`, consumed as module constants by
`src/podcast/script/pipeline.py`. Goal: format parity with NotebookLM, grounded in deep web
research and validated per format by a full judged bakeoff (the hybrid-v2 methodology).

User decisions (2026-07-11):
- **Scope**: all four formats; Deep Dive becomes one named format among four.
- **Validation**: full bakeoff per new format (2-3 candidate prompts each for Brief, Debate,
  Critique, judged head-to-head). Deep Dive keeps hybrid-v2 unchanged.
- **Debate hosts**: configured hosts (Alex & Maya) take genuinely opposing stances assigned
  at the outline stage; no moderator, no new host config.
- **Brief**: solo narrator (NotebookLM parity — Brief is NotebookLM's only single-speaker
  format), default the guide host (Maya).

## Research findings (6-agent web workflow, 695k tokens; full synthesis in task output)

**Headline: nobody has extracted Google's per-format prompts.** Every leak-repo "NotebookLM
prompt" is the chat/RAG prompt; all reconstructions (Hennig, Baoyu, Geller/Willison gist,
Rettberg empty-PDF probe arXiv:2511.08654) target the 2024 Deep Dive. Best-supported
inference [from Raiza Martin interview, latent.space]: one shared pipeline where the format
picker swaps the structural prompt section + voice casting, keeping grounding scaffolding —
exactly the architecture below.

Per-format observed behavior (help center + hands-on reports):
- **Brief**: ONE speaker, under 2 min, "just the essentials"; format choice (not the length
  slider) is the real compression control. No one has probed its internal structure.
- **Debate**: two hosts, formal back-and-forth; hosts derive the motion themselves from the
  source; **no verdict** — tension handed to the listener (~6:51 observed). Documented
  failure mode: "hosts are too agreeable" (MakeUseOf).
- **Critique**: two hosts, constructive tutor register ("there's some good stuff here, and,
  well, areas we can refine" — XDA transcript); targets authored drafts; covers gaps,
  clarity, actionable suggestions. Key tension: the Deep Dive base mandates neutrality +
  enthusiasm — a critique format must explicitly override both or it regresses to
  praise-heavy summary. **No OSS project ships a first-class critique format** (verified
  negative) — the rubric-review→dialogue-ification recipe below is novel.

Strongest transferable prompt sources (verbatim text captured in synthesis):
- OSS format mechanisms: PDF2Audio (per-format prompt dict — the whole format delta fits in
  1-3 sentences: persona + speaker count + word target), podcastfy (style variables +
  "Academic Debate" config), lfnovo/podcast-creator & open-notebook (format lives in the
  OUTLINE stage + personas; solo guard: "This is a SOLO podcast with only ONE speaker. Do
  NOT invent or add any other speakers."), SurfSense (duration→words→segment arithmetic),
  open-notebooklm (schema-bounded shortness: turn-count + 100-char line caps).
- Debate: ucl-dark/llm_debate debater prompts (per-round scaffolds: opening-with-context →
  list-opponent's-flaws rebuttal → answer-then-counter; "do not concede"; reward new
  arguments, penalize repetition), DEBATunE (enumerate arguments first, thread each),
  anti-sycophancy literature (Du et al. stubbornness; arXiv:2311.17371 agreement-intensity;
  arXiv:2305.19118 — pure contrarianism degrades quality, modest tit-for-tat wins;
  arXiv:2502.08788 — single structured pass ≈ multi-agent debate, so no agent machinery).
- Critique: Sakana AI-Scientist reviewer (rubric-forced output, "be specific to your current
  paper", reflection round, if-unsure-reject calibration), Liang et al. quota-forcing ("List
  4 key reasons… >=2 sub bullet points… in painstaking details"), CriticGPT
  (arXiv:2407.00215 — LLM critics hallucinate flaws → anchor every criticism to source
  text, drop unsupported ones), feedback-sycophancy (arXiv:2310.13548 — frame material as
  third-party), praise-sandwich evidence (arXiv:2502.12842).

## Design

### 1. Format registry — `src/podcast/script/formats.py` (NEW)

```python
@dataclass(frozen=True)
class FormatSpec:
    key: str                  # "deep-dive" | "brief" | "debate" | "critique"
    label: str                # "Deep Dive"
    description: str          # one-liner for CLI listing
    speakers: int             # 1 (brief) or 2
    system_prompt: str
    outline_brief: str
    polish_brief: str
    opening_position: str
    continuing_position: str
    final_position: str
    default_minutes: int      # brief=2, debate=7, critique=9, deep-dive=script.default_minutes(10)
    segment_range: tuple[int, int]   # brief (1,2); debate (4,7); others (3,6)
    length_mode: str          # "target" (extend or trim) | "ceiling" (trim only — brief)

FORMATS: dict[str, FormatSpec]
def resolve(key: str) -> FormatSpec   # error message lists valid keys
```

Prompt composition: shared invariant core (FORMAT FOR AUDIO rules, delivery-note grammar,
GROUNDING/attribution hygiene, host-label rules, listener framing) + per-format sections
(show concept, host dynamics, conversation engine, arc). **The composed deep-dive prompt
must be byte-identical to today's `SYSTEM_PROMPT`** (hybrid-v2 is validated property; a test
pins this). Research confirms hybrid-v2 already contains both extracted Deep Dive gems
(memorable-detail rule, closing reflective question) — no Deep Dive changes.

### 2. Per-format prompt content (bakeoff candidates seeded from research)

- **Brief** (solo, ~2 min ≈ 300 words, 1-2 segments): solo guard adapted from open-notebook;
  hook → 3-5 key takeaways → one actionable wrap line; word-budget arithmetic (SurfSense
  pattern), not adjectives; ban generic intros; keep light disfluencies + delivery notes but
  reduce density (banter/callback rules dropped — they presuppose two hosts); ensure_length
  acts as ceiling only.
- **Debate** (two hosts, ~7 min): outline derives a **motion** (actor+action+scope) from the
  source's genuine fault line — contradiction/tradeoff/prediction, never settled fact
  (false-balance guard); enumerates 3-4 arguments per side, threads each into a segment
  (DEBATunE); arc = cold open stating motion + stance handshake → openings with context →
  per-argument clash rounds (position hints carry the per-round scaffolds) → closings → no
  verdict. System prompt: assertive-but-scholarly; hold your stance, concede only narrow
  points; reward new arguments, penalize repetition; drop agreement affirmations
  ("Right"/"Exactly") and enthusiastic filler for this format; every claim source-anchored.
  ensure_length extends by adding an argument thread, never by padding rounds. Polish guard:
  "preserve the adversarial register; do not insert agreement or convergence."
- **Critique** (two hosts, ~9 min): **new pre-outline rubric stage** — structured review
  (Strengths / Weaknesses / Gaps / Assumptions / Questions / Actionable suggestions) as a
  pydantic model via `complete_structured`, with quotas (≥3 weaknesses, ≥2 sub-points each)
  and the grounding rule "every criticism anchors to quoted/paraphrased source content; if
  unsupported, drop it", plus one reflection round (Sakana). The outline stage
  dialogue-ifies this review. Roles: Maya = lead reviewer, Alex = clarifier/steelman ("is
  that fair?"). System prompt explicitly overrides the base neutrality + enthusiasm rules;
  third-party framing ("this document"); no praise sandwich; "specific to this document, no
  generic comments"; constructive-tutor register (XDA ritual as opening flavor).

### 3. Pipeline threading — `src/podcast/script/pipeline.py`

- `build_outline` / `write_dialogue` / `polish_dialogue` / `ensure_length` take the resolved
  `FormatSpec` (from `config.script.format`) instead of importing module constants; segment
  count range and word budget come from the spec.
- Solo path: `speakers == 1` → hosts brief lists only the solo host; dialogue schema's
  speaker enum is that one name. Downstream (synthesize/assemble/SoulX) already handles a
  transcript whose turns share one speaker; verify SoulX single-speaker input in tests + a
  real render.
- Debate stances: `Outline` model (`src/podcast/script/models.py`) gains optional
  `host_angles: dict[str, str]` (default empty) — outline LLM fills it for debate;
  `_hosts_brief` appends each host's stance. Critique rubric: new `CritiqueReview` model +
  stage function, gated by `format.key == "critique"`.

### 4. Config + CLI + provenance

- `ScriptSettings.format: str = "deep-dive"` (validator via `formats.resolve`);
  `ScriptSettings.solo_host: str | None = None` — validated against host names; `None` →
  first configured host; document `solo_host = "Maya"` (the guide) as the recommended
  setting and use it in the user-facing examples.
- `--format/-f` on `generate` and `create` (`src/podcast/cli/app.py`); `podcast formats`
  list command (pattern: existing `engines` command).
- Duration precedence: explicit `-d` > `FormatSpec.default_minutes` (deep-dive's default
  stays `script.default_minutes`, so existing behavior is unchanged).
- Episode front matter (`src/podcast/script/markdown.py`) records `format:`.
- Out of scope (noted for later): free-text `--instructions` customize block (research
  pattern: separate appended block, never merged into the format prompt); per-format voice
  casting.

### 5. Bakeoff workflows (full bakeoff × 3 new formats)

Per format (Brief, Debate, Critique), reuse the hybrid-v2 methodology:
1. 2-3 candidate prompts seeded from different research camps (e.g. Debate:
   persona-driven vs round-scaffold-driven vs hybrid) + a per-format rubric built from the
   observed-behavior findings (help-center definitions, XDA transcripts, failure modes).
2. Generation workflow: one script per candidate from `~/Downloads/RAG.md`.
   (Critique bakeoff judges rubric-stage + dialogue together.)
3. Judge panel, 4 lenses: format fidelity vs rubric, naturalness ear, faithfulness,
   listener preference → winner.
4. One refinement round on judge feedback → head-to-head verify.
5. Winner's text lands in `formats.py`; artifacts to `episodes/format-bakeoff-<format>/`
   (gitignored), like `rag-deep-dive/`.

### 6. Tests, docs

- `tests/test_formats.py` (NEW): registry completeness; resolve() error; deep-dive
  byte-equality pin; per-format invariants (solo guard present in brief, no bracketed cues,
  affirmation ban in debate, neutrality override in critique).
- `tests/test_pipeline.py`: FormatSpec threading; solo speaker enum; debate `host_angles`
  into hosts brief; critique rubric stage with fake provider; ensure_length ceiling mode.
- `tests/test_app.py`: `--format` flag; `formats` command; front-matter round-trip; solo
  episode through synthesize (incl. SoulX fake).
- ADR 0013 (formats feature + bakeoff provenance); README formats section.

## Verification

- `uv run pytest` green; ruff/mypy/pyright/pre-commit/CI gates green.
- Deep-dive regression: byte-equality test + one control generation (unchanged output path).
- End-to-end per format: `podcast create ~/Downloads/RAG.md --format brief|debate|critique`
  renders listenable episodes (SoulX; brief also on qwen3); user listening check.
- Bakeoff judge reports archived per format.

## Execution order

1. Refactor first, behavior-neutral: `formats.py` registry + pipeline threading + config/CLI
   + tests, deep-dive byte-identical (green gates, small PR-able unit).
2. Brief (solo path + placeholder prompt) → Brief bakeoff → wire winner.
3. Debate (host_angles + scaffolds) → Debate bakeoff → wire winner.
4. Critique (rubric stage) → Critique bakeoff → wire winner.
5. End-to-end renders, ADR 0013, README, PR.
