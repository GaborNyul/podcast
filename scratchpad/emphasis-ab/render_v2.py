"""Post-guard verification clips (audition round 2): the shipped 'both' path only.

Renders the marked sentences through the CURRENT qwen3 engine and its `_treated`
guard: a span is CAPSed and clause-named only when uppercasing changes it and it
is at least 3 chars; all-caps, numeric, and shorter spans get no treatment.

The committed v2 WAVs predate that final rule — they were rendered under the
round-1 guard (all-caps spans still carried a clause), and hearing that clause
stress the wrong word or nothing is what drove the round-2 flip (commit d449e98).
Re-running this script exercises the final rule.

    uv run python scratchpad/emphasis-ab/render_v2.py
"""

from pathlib import Path

from podcast.config import AppConfig
from podcast.tts.qwen3 import Qwen3Engine

OUT = Path(__file__).resolve().parent / "out"

# Under the final rule all three spans get no treatment (S4/S6 carried a
# clause under the round-1 guard; S3 has been untreated since round 1).
SENTENCES = {
    "S3": "That's *it* — the entire fix was one line.",
    "S4": "Plain *RAG* retrieves neighbors of the question, not the answer.",
    "S6": "Turns out *AI* is doing the reranking here.",
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
