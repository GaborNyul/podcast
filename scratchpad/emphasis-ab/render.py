"""A/B listening demos for ADR 0014 word-level emphasis (final hardware validation).

Renders fixed RAG-flavored sentences through the PUBLIC engine APIs:
  - qwen3: four arms per sentence (plain / caps / instruct / both), voice vivian
  - soulx: one 3-line Alex/Maya mini-dialogue rendered marked vs stripped

Run from the repo root (one engine per process so the GPU holds one model):

    uv run python scratchpad/emphasis-ab/render.py qwen3
    uv run python scratchpad/emphasis-ab/render.py soulx
"""

import sys
from collections.abc import Callable
from pathlib import Path

from podcast import emphasis
from podcast.config import AppConfig
from podcast.tts.base import DialogueLine
from podcast.tts.qwen3 import Qwen3Engine
from podcast.tts.soulx import SoulXEngine

OUT = Path(__file__).resolve().parent / "out"

S1 = "The bottleneck wasn't retrieval — it was the *reranker* all along."
S2 = "*Not* every query needs retrieval."
S3 = "That's *it* — the entire fix was one line."
S4 = "Plain *RAG* retrieves neighbors of the question, not the answer."
S5 = "And that is why hybrid search wins. Wait — you are telling me the fix was *free*"

QWEN3_SENTENCES = {"S1": S1, "S2": S2, "S3": S3, "S4": S4}
QWEN3_VOICE = "vivian"


def instruct_clause(sentence: str) -> str:
    """The shipped clause wording, built via the public emphasis.spans API."""
    span_texts = emphasis.spans(sentence)
    if len(span_texts) != 1:
        raise ValueError(f"demo sentences carry exactly one span: {sentence!r}")
    return f"Put strong emphasis on the word '{span_texts[0]}'."


def qwen3_arms(sentence: str) -> dict[str, tuple[str, str]]:
    """arm -> (text handed to synthesize_line, delivery)."""
    return {
        # Control: no markup, no instruct.
        "plain": (emphasis.strip_markup(sentence), ""),
        # Pre-rendered CAPS: engine sees no spans, adds no clause.
        "caps": (emphasis.render_caps(sentence), ""),
        # Clause only: stripped text, clause rides the delivery/instruct channel.
        "instruct": (emphasis.strip_markup(sentence), instruct_clause(sentence)),
        # Shipped path: marked text, engine does CAPS + clause itself.
        "both": (sentence, ""),
    }


def render_qwen3() -> None:
    engine = Qwen3Engine(AppConfig())  # one loaded model reused for all 16 renders
    for key, sentence in QWEN3_SENTENCES.items():
        for arm, (text, delivery) in qwen3_arms(sentence).items():
            out = OUT / f"qwen3-{key}-{arm}.wav"
            print(f"[qwen3] {out.name}: text={text!r} delivery={delivery!r}", flush=True)
            engine.synthesize_line(text, QWEN3_VOICE, out, delivery=delivery)
            print(f"[qwen3] wrote {out}", flush=True)


DIALOGUE = [
    ("Alex", "So we spent the whole sprint blaming the vector index."),
    ("Maya", S1),  # line 2 = S1 (plain mid-sentence mark)
    ("Alex", S5),  # line 3 = S5 (emphasized unpunctuated FINAL word, final line)
]
VOICES = {"Alex": "alex", "Maya": "maya"}  # repo-shipped assets/voices/soulx refs


def render_soulx() -> None:
    engine = SoulXEngine(AppConfig())  # one loaded model reused for both passes
    arms: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("marked", lambda t: t),  # engine converts *spans* to stress tokens
        ("stripped", emphasis.strip_markup),  # control: identical words, no tokens
    )
    for tag, transform in arms:
        lines = [DialogueLine(speaker=s, text=transform(t)) for s, t in DIALOGUE]
        outs = [OUT / f"soulx-{tag}-{i}.wav" for i in (1, 2, 3)]
        print(f"[soulx] {tag}: {[line.text for line in lines]}", flush=True)
        engine.synthesize_dialogue(lines, VOICES, outs)
        print(f"[soulx] wrote {[str(p) for p in outs]}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("qwen3", "all"):
        render_qwen3()
    if which in ("soulx", "all"):
        render_soulx()
    print("done", flush=True)
