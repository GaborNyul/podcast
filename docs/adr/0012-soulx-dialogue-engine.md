# ADR 0012: SoulX-Podcast dialogue-native engine

Date: 2026-07-11
Status: Accepted

## Context

NotebookLM's engagement comes from rendering whole conversations, not isolated lines
(ADR 0003 reserved a slot for such engines; the ADR 0010/0011 research confirmed the
mechanism). A gfx1151 spike proved SoulX-Podcast-1.7B (Apache-2.0, Qwen3-1.7B backbone)
runs on this box at RTF ~2.8 with voices zero-shot-cloned from qwen3-minted references;
the user judged its output clearly better than per-line qwen3.

## Decision

- `DialogueEngine` protocol (`synthesize_dialogue(lines, voices, out_paths)`) joins
  `podcast.tts.base`; `_run_synthesize` branches on `EngineInfo.dialogue_native` and
  caches per whole dialogue (`dialogue-<hash>-<idx>.wav`) — any line edit re-renders the
  conversation, because every line's prosody depends on its predecessors. Tempo variants
  and assembly are shared with the per-line path.
- `SoulXEngine` drives the upstream inference source from a commit-pinned git checkout
  under `models_dir` (no PyPI package exists; PyPI deps live in the locked `soulx` extra).
  torchaudio 2.9's load/save are shimmed through soundfile (torchcodec is not ABI-safe
  against TheRock wheels); numba/librosa JIT is disabled (gfx1151 llvmlite segfault).
- Voices are clone references: `[tts.soulx_refs]` maps voice ids to a reference WAV plus
  sidecar `.txt` transcript. The user-validated pair ships in `assets/voices/soulx/`
  (minted with qwen3: the reference's register is the cloned voice's emotional baseline —
  a neutral reference reads flat with misplaced emphasis).
- Delivery notes map onto SoulX's documented paralinguistic tags by keyword
  (laugh/sigh/breath); everything else rides on dialogue context.

## Consequences

- `synthesize` on soulx re-renders whole conversations; the per-line cache advantage
  stays with kokoro/qwen3. Host `tempo` still applies (user-validated: male 1.05–1.10).
- First use clones the pinned source and downloads ~4 GB of weights.
- Up to 4 speakers (SoulX limit); more hosts fail loudly.
