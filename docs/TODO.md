# TODO

Planned follow-up work; items move into ADRs/specs when they start.

## 0. Define source code licensing (MIT, GNU, etc.)

The repository is public but has no license, which legally means all rights reserved.
Pick a license (MIT, Apache-2.0, GPL, ...), add a `LICENSE` file at the repo root, and
reference it from `README.md` and the project metadata in `pyproject.toml`.

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
