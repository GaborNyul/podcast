# podcast

Local-first, NotebookLM-style podcast generator: feed it documents, get a two-host
dialogue script you can edit, then a rendered audio episode — all on your own machine.

```
docs (txt/md/html/pdf/docx) ──▶ podcast generate ──▶ episodes/<slug>/script.md
                                                          │  (edit by hand, freely)
                                                          ▼
                                 podcast synthesize ──▶ episodes/<slug>/episode.mp3
```

- **Local by default** — script writing via Ollama, speech via local TTS models.
- **Editable contract** — `script.md` is the source of truth; hand edits survive, and
  the per-line segment cache means editing one line re-renders one line.
- **Hardware-aware** — Qwen3-TTS on an AMD Strix Halo GPU for quality, Kokoro-82M on
  CPU everywhere else.
- **Expressive delivery** — each line can carry a performance note
  (`**Maya [excited, leaning in]:** Get this...`); the qwen3 engine performs it via
  Qwen3-TTS instruction control, engines without that ability ignore it (ADR 0010).
- **Dialogue-native option** — the `soulx` engine (SoulX-Podcast-1.7B) renders the whole
  conversation in one pass, so every line reacts to the lines before it; voices are
  clones of the reference WAVs in `assets/voices/soulx/` (ADR 0012).

## Install

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.13, and `ffmpeg` on PATH.

```sh
git clone https://github.com/gabornyul/podcast && cd podcast
uv sync                 # core install: CPU synthesis (Kokoro), all LLM providers
uv run podcast doctor   # verify ffmpeg, workspace, and the configured engine
```

## Quickstart

```sh
# 1. Generate an editable script from documents (10 minutes by default)
uv run podcast generate paper.pdf notes.md -d 15

# 2. Open episodes/<slug>/script.md, tweak any lines you like — including the
#    [delivery notes] that steer tone and pace on the qwen3 engine

# 3. Render the audio (only edited lines are re-synthesized on re-runs)
uv run podcast synthesize

# ...or do both in one shot:
uv run podcast create paper.pdf -d 15 --engine kokoro --provider ollama

# ...and pick the conversation format (deep-dive is the default):
uv run podcast create paper.pdf --format debate
```

Useful commands: `podcast doctor` (environment checks), `podcast engines` /
`podcast voices` (what can this machine speak with), `podcast formats` (the
audio overview formats), `podcast config` (resolved configuration as JSON).

## Formats

NotebookLM-style audio overview formats (ADR 0013), selected with `--format`/`-f`
on `generate`/`create` or `format` under `[script]`:

| format      | speakers  | default length | what it is |
|-------------|-----------|----------------|------------|
| `deep-dive` | two hosts | ~10 min        | the default: a detailed, banter-rich exploration of the sources |
| `brief`     | solo      | ~2 min         | hook, the 2-4 essential points, one landing line — never padded |
| `debate`    | two hosts | ~7 min         | hosts argue opposing stances on the sources' real open question; no verdict |
| `critique`  | two hosts | ~9 min         | an anchored, constructive expert review of the material with concrete fixes |

Brief is narrated by one host — the first configured host unless `[script]
solo_host = "Maya"` picks another. Debate assigns each host a stance at the
outline stage; Critique runs a structured review pass over the sources before
outlining. Each new format's prompt was selected by a judged bakeoff
(`docs/adr/0013-audio-overview-formats.md`).

## Configuration

Layered: defaults ← `~/.config/podcast/config.toml` ← `./podcast.toml` ← environment
variables (`PODCAST_SECTION__KEY`). Example `podcast.toml`:

```toml
[llm]
provider = "ollama"                        # ollama | ollama-cloud | openai | anthropic | copilot | fake
model = "qwen3:30b-a3b-instruct-2507"      # omit to use the provider's default

[script]
default_minutes = 10
format = "deep-dive"                        # deep-dive | brief | debate | critique
solo_host = "Maya"                          # who narrates solo formats (brief)
[[script.hosts]]
name = "Alex"
gender = "male"
persona = "the companion (curious co-host): asks stake-bearing questions, pushes back"
[[script.hosts]]
name = "Maya"
gender = "female"
persona = "the guide (lead explainer): explains with vivid analogies and concrete examples"
style = "Speak at a fast, energetic pace."  # baseline instruct for delivery-capable engines
tempo = 1.1                                 # pitch-preserving speed-up of this host's lines

[tts]
engine = "qwen3"                           # qwen3 (GPU) | kokoro (CPU) | soulx (GPU, dialogue-native)
qwen3_temperature = 0.8                    # sampling; lower reads robotic (ADR 0011)
[tts.voices]                               # optional per-speaker voice overrides
                                           # (engine-specific ids — swap when switching engines)
Alex = "Ryan"

[tts.calibration]                          # measured rendered-wpm / 150 per engine
qwen3 = 0.87
kokoro = 1.02

[audio]
pause_min_ms = 200
pause_max_ms = 1000
mp3_bitrate = "192k"
```

### LLM providers

| Provider | Needs | Default model |
|---|---|---|
| `ollama` | local Ollama at `localhost:11434` | `qwen3:30b-a3b-instruct-2507` |
| `ollama-cloud` | `PODCAST_LLM__API_KEY` | `qwen3-coder:480b-cloud` |
| `openai` | `PODCAST_LLM__API_KEY` | `gpt-5` |
| `anthropic` | `PODCAST_LLM__API_KEY` or `ANTHROPIC_API_KEY` | `claude-opus-4-8` |
| `copilot` | GitHub account with Copilot (first run opens a device-flow login) | `gpt-4o` |
| `fake` | nothing — deterministic offline scripts for testing | — |

## GPU synthesis on AMD Strix Halo (gfx1151)

The `qwen3` engine runs Qwen3-TTS-12Hz-1.7B on the iGPU via AMD's **TheRock**
nightly PyTorch wheels — stock ROCm wheels do not support gfx1151.

```sh
uv sync --extra qwen3     # pulls torch/torchaudio + ROCm runtime from TheRock
uv run podcast doctor     # must show the GPU under "qwen3 engine"
```

Rules of the road (learned the hard way — see `docs/adr/0004`):

- **Never set `HSA_OVERRIDE_GFX_VERSION`.** gfx1151 must not masquerade as gfx1100;
  `podcast doctor` fails loudly if it is set.
- The wheels are pinned to a known-good nightly
  (`torch==2.9.1+rocm7.13.0a20260501` from
  `https://rocm.nightlies.amd.com/v2/gfx1151/`). Nightlies get pruned upstream —
  if the pinned build disappears, bump the pins in `pyproject.toml` to a current
  date and re-run `uv lock`.
- Conv-heavy models can be *slower* on this GPU than on CPU; that is why engines
  declare their own preferred device and Kokoro stays CPU-only.

### Hardware verification checklist (after merging / on the target box)

1. `uv sync --extra qwen3`
2. `uv run podcast doctor` → all rows green, `qwen3 engine` names the GPU
3. `uv run pytest -m integration -k qwen3 -s` → prints the RTF benchmark and a
   suggested `[tts.calibration] qwen3 = <rendered_wpm / 150>`; record it in config
4. `uv run podcast create <real docs> --engine qwen3 -d 15` → episode length within
   ±15 % of the target

## Development

Standards: [docs/python_development_standards_v3.md](docs/python_development_standards_v3.md)
(ruff + mypy/pyright strict, 100 % coverage, hypothesis, JSON-schema contracts).
Design: [docs/superpowers/specs/](docs/superpowers/specs/) and [docs/adr/](docs/adr/).

```sh
uv run python scripts/pre_ticket.py <id>    # clean-baseline gate before starting work
uv run python scripts/post_ticket.py <id>   # all gates + lockfile check after work
uv run python scripts/adversarial_review.py # bounded reviewer/fixer loop (pre-PR)
uv run pre-commit run --all-files
uv run pytest -m performance --timeout=60   # nightly perf suite
uv run pytest -m integration                # real-model tests (downloads weights)
```

Notes on the pre-commit setup: mypy, pyright, pip-audit, and pip-licenses run as
`uv run` local hooks (instead of the upstream mirrors) so they check against the
project's locked dependencies rather than an empty isolated environment.

The workspace artifacts (`sources.json`, `outline.json`, `transcript.json`) are a
public contract validated by JSON Schemas in [schemas/](schemas/); changing the
models requires deliberately regenerating those files (see `tests/test_contracts.py`).

## License

Licensed under the **GNU Affero General Public License v3.0 or later (AGPLv3+)** —
see [LICENSE](LICENSE). In short: any distributed or **network-hosted** derivative must
also be released under the AGPLv3, with source offered to its users. Podcasts you
generate are your own — program output is not a derivative of the program.

Copyright (C) 2026 Gabor Nyul
