# TODO

Planned follow-up work; items move into ADRs/specs when they start.

## 1. Per-format default minutes configurable from podcast.toml

Each format's default length (`FormatSpec.default_minutes`: brief 2, debate 7, critique 9;
deep-dive defers to `script.default_minutes`) is currently hard-coded in
`script/formats.py`. Make it overridable per format from configuration — podcast.toml and
every other layer (user config, `PODCAST_*` env vars):

1. Add e.g. `[script.format_minutes]` (`brief = 3`, `debate = 10`, ...) to
   `ScriptSettings`, validated against known format keys.
2. Resolution order in `_episode_minutes` becomes: explicit `-d` > config override for the
   selected format > `FormatSpec.default_minutes` > `script.default_minutes`.
3. `podcast formats` table shows the resolved (overridden) length, marking overrides.
4. Tests: layering (project over user over defaults), env-var form, unknown-key rejection.

## 2. Hennig-style empirical refinement of the new formats

The brief/debate/critique prompts (ADR 0013) were selected by judged bakeoff on a single
source (RAG.md), judged on *scripts*. NotebookLM's formats were reverse-engineered the other
way around — from what listeners actually hear. Close that loop:

1. Generate a corpus: several episodes per format (brief, debate, critique) across
   genuinely different topics — technical paper, news analysis, opinion essay, product
   doc — so per-format rituals separate from per-topic content.
2. Run STT (e.g. whisper) over the rendered audio — analyzing what survives TTS, pacing,
   and the polish pass, not the pre-render script.
3. Analyze transcripts the way Nicole Hennig reverse-engineered NotebookLM
   (https://nicolehennig.com/notebooklm-reverse-engineering-the-system-prompt-for-audio-overviews/):
   catalog recurring opening/closing rituals, transition phrases, affirmation and filler
   distribution, structure per format, per-host role fidelity — and compare against real
   NotebookLM output in the same formats on the same sources.
4. Feed the divergence catalog back: refine or re-bake the format prompts (the bakeoff
   workflow is parameterized and reusable; rubrics live in the ADR 0013 provenance and
   `episodes/format-bakeoff-*/`), and pin the newly validated texts in
   `tests/test_formats.py`.

## 3. Clone the original NotebookLM voices for qwen3 and SoulX

Replace the qwen3-minted references with voices cloned from real NotebookLM audio:

1. Generate NotebookLM Audio Overviews and extract clean, single-speaker snippets of the
   male and female hosts (a few seconds each, low music/overlap; pick emotionally lively
   passages — the reference's register IS the cloned voice's emotional baseline).
2. SoulX path: reference WAV (≤15 s) + sidecar transcript per host into
   `assets/voices/soulx/`, wired via `[tts.soulx_refs]`; re-run the A/B listening check
   against the current alex/maya references.
3. qwen3 path: the Base checkpoint (Qwen3-TTS-12Hz-1.7B-Base) does 3-second zero-shot
   cloning but supports NO `instruct` — evaluate whether losing per-line delivery notes
   is worth the timbre, or keep CustomVoice+instruct for qwen3 and clone only for SoulX.
4. Check the licensing/ToS question of cloning NotebookLM's voices before shipping
   anything beyond local experiments.

## 4. Sweep qwen3_temperature for naturalness (sampling variance dwarfs treatment effects)

The 2026-07-15 emphasis audition (ADR 0014, `scratchpad/emphasis-ab/MANIFEST.md`) found
that two clips rendered from *byte-identical* inputs (S3-plain vs S3-both-v2: same text,
no instruct) were judged "most natural" and "overstimulated, not natural" — at the shipped
`qwen3_temperature = 0.8`, run-to-run variance is larger than most deliberate treatment
effects. Find the temperature that maximizes natural delivery:

1. Sweep e.g. 0.5 / 0.6 / 0.7 / 0.8 / 0.9 (hold top_p 0.9, repetition_penalty 1.05 fixed
   first; ADR 0011 chose 0.8 because lower "reads robotic" — re-test that claim per temp).
2. N ≥ 3 repeat takes per temperature on 2-3 fixed sentences (reuse the
   `scratchpad/emphasis-ab/` harness pattern) — single clips proved unable to separate
   treatment from noise; judge take-to-take consistency as well as naturalness.
3. Blind the listening (shuffled filenames), pick the winner, update `qwen3_temperature`
   default + `[tts]` docs, and re-run one emphasis A/B round at the new temp to confirm
   the ADR 0014 verdicts still hold (CAPS+clause wins, guards unchanged).
