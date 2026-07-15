"""Post-guard verification clips (audition round 2): the shipped 'both' path only.

Renders the marked sentences through the CURRENT qwen3 engine, which now applies
the per-span guards (short non-all-caps spans untreated; all-caps spans clause-only).

    uv run python scratchpad/emphasis-ab/render_v2.py
"""

from pathlib import Path

from podcast.config import AppConfig
from podcast.tts.qwen3 import Qwen3Engine

OUT = Path(__file__).resolve().parent / "out"

SENTENCES = {
    "S3": "That's *it* — the entire fix was one line.",  # guard: no CAPS, no clause
    "S4": "Plain *RAG* retrieves neighbors of the question, not the answer.",  # clause only
    "S6": "Turns out *AI* is doing the reranking here.",  # 2-char all-caps: clause fires
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = Qwen3Engine(AppConfig())
    for key, sentence in SENTENCES.items():
        out = OUT / f"qwen3-{key}-both-v2.wav"
        print(f"[qwen3] {out.name}: {sentence!r}", flush=True)
        engine.synthesize_line(sentence, "vivian", out, delivery="")
        print(f"[qwen3] wrote {out}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
