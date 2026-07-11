# ADR 0013: Audio overview formats (Brief / Deep Dive / Debate / Critique)

Date: 2026-07-11
Status: Accepted

## Context

NotebookLM ships four Audio Overview formats; this generator hard-wires one — the
hybrid-v2 "Deep Dive" prompt (ADR 0009) exported as module constants from
`script/prompts.py`. A six-agent web research pass established that no Google per-format
prompt has ever been extracted, and that the best-supported architecture (team interviews,
OSS clones) is one shared pipeline whose format picker swaps the structural prompt section:
persona/speaker count/arc — not new machinery. Two findings shaped the design: Brief is
NotebookLM's only single-speaker format, and no OSS project ships a first-class Critique
format (rubric-review → dialogue-ification is novel). Multi-agent debate literature
(arXiv:2502.08788) says a single structured pass captures most of a debate's value, so
formats stay prompt-level, not agent-level.

## Decision

- `script/formats.py` owns a `FormatSpec` registry (`deep-dive`, `brief`, `debate`,
  `critique`): per-format system prompt, outline/polish briefs, position hints, speaker
  count, default minutes, segment range, and length mode. Prompts are composed from a
  shared invariant core (audio rules, delivery-note grammar, grounding) plus per-format
  sections; the composed deep-dive prompt is **byte-identical** to the validated hybrid-v2
  text, pinned by a test.
- The pipeline (`build_outline`/`write_dialogue`/`polish_dialogue`/`ensure_length`)
  consumes the resolved `FormatSpec` instead of prompt constants. `length_mode="ceiling"`
  (Brief) trims but never extends.
- Brief is solo: the dialogue schema's speaker enum shrinks to `script.solo_host`
  (default: first configured host; `"Maya"` recommended), and the prompt carries a solo
  guard — downstream synthesis already handles single-speaker transcripts.
- Debate assigns stances at the outline stage: `Outline.host_angles` (optional, empty for
  other formats) maps each host to a stance on a motion derived from the source's genuine
  fault line; the hosts brief carries the stances into every dialogue request. No
  moderator, no verdict.
- Critique adds a pre-outline rubric stage: a structured review (strengths, weaknesses,
  gaps, assumptions, questions, suggestions — with quotas and a drop-unsupported-criticism
  rule) that the outline then dialogue-ifies. The critique system prompt explicitly
  overrides the base neutrality and enthusiasm rules.
- Selection: `script.format` in config, `--format/-f` on `generate`/`create`, a `formats`
  listing command, and a `format:` front-matter field. Explicit `-d` beats the format's
  default minutes; deep-dive keeps `script.default_minutes`.
- Each new format's prompt text is selected by a judged bakeoff (candidates seeded from
  the research corpus, 4 judge lenses, one refinement round), the ADR 0009 methodology.
  Results (2026-07-11, ~1.5M tokens across three 15-agent workflows, RAG.md source):
  - **Brief**: the radio-news-craft candidate ("the Bulletin") won 3 of 4 lenses
    (rank sums C=1, A=6, B=5); its refinement lost the head-to-head 1/3 — original ships.
  - **Debate**: the persona-driven "fire vs ice" sparring-partners candidate won
    (B=2, A=5, C=5); the verification round was cut short by an API budget limit,
    so the un-refined winner ships.
  - **Critique**: the rubric-forward audit candidate won (B=1, A=3, C=8) and its
    refinement won the head-to-head 2/3 — refined text ships.
  A 45-agent adversarial review of the refactor confirmed 9 findings (1 major: no
  end-to-end test covered the critique review stage wiring), all fixed: volunteered
  `host_angles` are cleared outside debate, stance keys are normalized
  case-insensitively, debate/critique trim to their two roles via
  `FormatSpec.speakers`, and the CLI help/table text was corrected.

## Consequences

- Existing episodes and configs are untouched: `deep-dive` is the default and its prompt
  bytes are unchanged.
- Brief renders fastest and is the natural smoke-test format; its ceiling length mode
  means short outputs are accepted rather than padded.
- Debate drops the affirmation glue the other formats rely on — polish must preserve the
  adversarial register, so its polish brief diverges from the shared one.
- Critique costs one extra LLM call (rubric stage) per episode.
- The `--instructions` free-text customize block and per-format voice casting are
  deliberately out of scope (recorded in the design spec).
