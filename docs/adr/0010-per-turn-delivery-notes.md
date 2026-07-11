# ADR 0010: Per-turn delivery notes drive TTS emotion (qwen3 `instruct`)

Date: 2026-07-11
Status: Accepted

## Context

Rendered episodes sound monotone next to NotebookLM. Research into how NotebookLM and its
open-source clones achieve expressive audio found two mechanisms: whole-dialogue-context
synthesis (dialogue-native models; reserved for later by ADR 0003) and a per-line emotion
channel from the script LLM into the voice engine (Meta NotebookLlama's engine-aware tag
rewriting; OpenAI's `instructions` parameter; ElevenLabs v3 audio tags). The decisive local
fact: the Qwen3-TTS-12Hz-1.7B-CustomVoice checkpoint this project already runs accepts an
optional natural-language `instruct` argument per utterance ("Speak in an excited tone") —
verified in the installed `qwen-tts` package — and the engine simply never passed it.
Kokoro-82M has no comparable input; its author documents emotion as a training-data gap that
inference cannot fill.

## Decision

- `Turn` gains an optional `delivery` field: a short performance note (tone, pace,
  emotional register), written by the LLM alongside each line and never spoken. The system
  prompt's FORMAT FOR AUDIO section directs the notes; spoken text stays cue-free.
- script.md carries the note in the speaker token — `**Host [excited, leaning in]:** text` —
  so hand editors can tune performances line by line. The token grammar is kept unambiguous
  from both sides: notes are whitespace-normalized and stripped of `[`/`]`/`:` (the
  line-grammar characters) on both write and parse, and host names may not contain those
  characters either — `podcast.config` rejects them at load and the parser rejects them in
  front matter. A bare `**Host:**` line stays valid, so pre-existing scripts parse unchanged.
- Script excerpts quoted back to the LLM (segment context, length repair) use the same
  notation; the system prompt explains it and says the speaker field is only ever the bare
  host name, `_resolve_speaker` tolerates a model echoing a `Name [note]` token anyway, and
  the length-repair prompt states that notes never count toward the word target.
- `TTSEngine.synthesize_line` gains a keyword-only `delivery` parameter and `EngineInfo`
  a `supports_delivery` capability flag. qwen3 forwards non-blank notes as `instruct`
  (blank means no instruction); kokoro declares no support and ignores them.
- The segment cache key includes delivery, but the CLI blanks the note for engines that
  do not support it — so on kokoro, editing a note re-renders nothing, while on qwen3 an
  edited note re-renders exactly that line.

## Consequences

- Emotion control with zero new dependencies on the existing hardware-verified engine; the
  0.6B CustomVoice checkpoint would silently ignore `instruct`, which is one more reason the
  engine pins the 1.7B model.
- The cache key gained a field, so segments rendered before this change re-render once.
- The transcript artifact schema gained the optional `delivery` property (additive;
  `schemas/transcript.schema.json` regenerated deliberately).
- A future dialogue-native engine (ADR 0003) can consume the same notes or ignore them in
  favor of conversation context.
