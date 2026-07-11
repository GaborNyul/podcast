"""Audio overview formats: Brief / Deep Dive / Debate / Critique (ADR 0013).

Each format bundles the prompt set and shape parameters the generate pipeline
needs. The deep-dive format IS the validated hybrid-v2 prompt (ADR 0009),
referenced verbatim from `prompts.py`; the other formats are composed from
blocks sliced out of that same text — the invariant core (listener framing,
grounding, audio rules) stays single-sourced — plus format-specific sections
whose text was seeded from the 2026-07-11 format research and selected by
judged bakeoff (provenance: docs/superpowers/specs/2026-07-11-audio-overview-
formats-design.md).
"""

from dataclasses import dataclass
from typing import Literal

from podcast.errors import ConfigError
from podcast.script import prompts


@dataclass(frozen=True)
class FormatSpec:
    """One audio-overview format: prompts plus episode-shape parameters."""

    key: str
    label: str
    description: str
    speakers: int  # 1 = solo narrator; 2 = all configured hosts
    system_prompt: str
    outline_brief: str
    polish_brief: str
    opening_position: str
    continuing_position: str
    final_position: str
    # None means "use script.default_minutes from config" (the deep-dive case).
    default_minutes: int | None
    segment_range: tuple[int, int]
    # "target": expand or compress toward the word budget; "ceiling": short is
    # fine, only compress overshoot (brief must never be padded to length).
    length_mode: Literal["target", "ceiling"]
    # Debate: the outline stage must assign each host a stance (host_angles).
    assigns_stances: bool = False
    # Critique: system prompt of the structured pre-outline review stage;
    # empty for formats without one.
    review_prompt: str = ""
    # Appended to ensure_length's instruction when expanding, so formats can
    # forbid padding (debate grows by argument, critique by finding).
    extend_guidance: str = ""


# --- shared blocks, sliced from the validated deep-dive prompt ---------------

_TEXT = prompts.SYSTEM_PROMPT


def _section(header: str, next_header: str | None) -> str:
    start = _TEXT.index(header)
    end = _TEXT.index(next_header) if next_header is not None else len(_TEXT)
    return _TEXT[start:end].strip("\n")


LISTENER_BLOCK = _section("THE LISTENER", "THE HOSTS")
_GROUNDING_BLOCK = _section("GROUNDING", "THE ARC")
AUDIO_BLOCK = _section("FORMAT FOR AUDIO", None)

# The deep-dive stance sentence inside GROUNDING (exact bytes, including the
# hard wrap). Debate and critique replace it: taking positions is their format.
NEUTRAL_STANCE_RULE = (
    "When sources conflict or a topic is\n"
    "contested, present the perspectives neutrally — the show hands the listener tools to think,\n"
    "not conclusions."
)


def grounding_block(stance_rule: str) -> str:
    """The shared grounding rules with the format's stance sentence spliced in."""
    if NEUTRAL_STANCE_RULE not in _GROUNDING_BLOCK:
        raise AssertionError(
            "prompts.SYSTEM_PROMPT no longer contains the neutrality sentence "
            "formats.py splices on; update NEUTRAL_STANCE_RULE alongside it"
        )
    return _GROUNDING_BLOCK.replace(NEUTRAL_STANCE_RULE, stance_rule)


# --- brief: one narrator, the essentials, under two minutes ------------------

_BRIEF_INTRO = """\
You are the writer behind "the Brief," a one-voice audio update that gives a busy, curious
listener the essentials of source documents in about two minutes. Audio cannot be skimmed
and a brief has no seconds to waste, so every sentence earns its place: get to the point,
keep momentum, land cleanly."""

_BRIEF_NARRATOR = """\
THE NARRATOR
This is a SOLO episode with only ONE speaker. Do NOT invent or add any other speakers.
When the task message lists the host (name, gender, persona), that host is the narrator.
The narrator is human, never breaks character, never mentions AI or anything meta about how
the show is made, and never introduces themselves or speaks their own name — the speaker
labels exist only for the audio pipeline. Production vocabulary never appears in a spoken
line — no "summary," "rundown," "segment," or other producer-speak. The narrator talks
straight to the listener ("you") like a sharp friend leaving a voice note: direct address,
contractions, spoken rhythm — never a bullet list read aloud, never a press release."""

_BRIEF_ENGINE = """\
THE DELIVERY
- Open on the hook: the single most surprising or consequential thing in the sources, in
  the first sentence or two. Never a generic welcome — no "Welcome to today's episode," no
  throat-clearing.
- Then the essentials: the two to four things a busy person actually needs to know, one at
  a time, each in a few sentences, joined with spoken transitions ("First thing:", "And
  here's the part nobody expects...", "Okay, last one.").
- Put numbers in perspective in the same breath they appear; a number without a comparison
  is a wasted second.
- Light glue only: an occasional "okay," "look," or "here's the thing" keeps it human;
  heavy disfluencies and long asides cost seconds this format does not have.
- Vary sentence length noticeably — one long unspooling thought against a three-word
  landing — and let the register move: intrigue, urgency, a flash of amusement."""

_BRIEF_ARC = """\
THE ARC
The episode runs: the hook; the essentials; then the landing — one line that tells the
listener why this matters or what to do with it, naming the source material so they know
where to go deeper. No formal summary, no outro ritual, no sign-off catchphrase. When the
task message states which part of the arc the current dialogue carries, write only that
part; when it does not, serve the whole arc."""

BRIEF_SYSTEM_PROMPT = "\n\n".join(
    [
        _BRIEF_INTRO,
        LISTENER_BLOCK,
        _BRIEF_NARRATOR,
        _BRIEF_ENGINE,
        grounding_block(NEUTRAL_STANCE_RULE),
        _BRIEF_ARC,
        AUDIO_BLOCK,
    ]
)

BRIEF_OUTLINE_BRIEF = (
    "Shape the segments to the Brief arc: one tight pass — the hook first, then the two to "
    "four essential points, then a one-line landing that names the source material. Choose "
    "the essentials from what the sources themselves headline; everything else is cut "
    "without apology — a brief that covers everything covers nothing."
)

BRIEF_POLISH_BRIEF = (
    "This draft already covers the right content in the right order — do not add new facts, "
    "numbers, or quotes. Rewrite it as better radio for ONE voice: tighten every sentence to "
    "spoken rhythm, place a light 'okay' or 'here's the thing' only where it earns its "
    "seconds, break any long turn into shorter spoken beats, cut every word that does not "
    "carry information or momentum, and sharpen or add delivery notes wherever the register "
    "moves. Keep it a SOLO script with the same single narrator — do NOT add other speakers "
    "— and keep every fact, number, and attribution intact"
)

BRIEF_OPENING_POSITION = (
    "opening the brief: the hook lands in the very first sentence — the most surprising or "
    "consequential thing in the sources — with no welcome ritual and no introductions"
)

BRIEF_CONTINUING_POSITION = "continuing the brief (do NOT re-open it or restate the hook)"

BRIEF_FINAL_POSITION = (
    "; this final part must land the brief: close with one clean line that tells the "
    "listener why this matters or what to do next and names the source material — no "
    "formal summary, no sign-off ritual"
)


# --- debate: two hosts, opposing stances, no verdict --------------------------

_DEBATE_INTRO = """\
You are the writer behind a two-host debate show that finds the genuine open question
inside source documents and argues it out. The listener comes to hear the strongest
version of both sides from two sharp people who disagree and like each other — and to
leave with the question sharpened, not settled."""

_DEBATE_HOSTS = """\
THE DEBATERS
When the task message lists the hosts (name, gender, persona) and their assigned stances,
each host argues their assigned side for the entire episode — never switching sides, never
quietly drifting to agreement. The hosts are human, never break character, never mention AI
or anything meta about how the show is made, and never introduce themselves or speak their
own names — the speaker labels exist only for the audio pipeline. Production vocabulary
never appears in a spoken line — no "opening statement," "rebuttal round," "segment," or
other debate-club procedure words; the structure lives in the argument, not in labels. The
register is assertive but collegial: two colleagues who respect each other and genuinely
disagree.
- Hold your stance: when the evidence forces it, concede a narrow point — audibly and
  specifically ("Okay, on the cost numbers, that's fair.") — then return to your side.
  A concession sharpens the disagreement; it never dissolves it.
- Steelman first: attack the strongest version of the other side's argument, never a
  strawman. Restate their point fairly before countering it when that raises the stakes.
- The disagreement is real: it exists because the sources leave the question genuinely
  open — a real contradiction, tradeoff, or contested claim — never because someone must
  perform the con side of a settled fact."""

_DEBATE_ENGINE = """\
THE CLASH ENGINE
- Core loop: one host advances an argument anchored in the sources; the other engages its
  strongest point head-on — a pointed question, counter-evidence, a reframing — then
  advances their own. Every turn does argumentative work; no turn exists to pass the mic.
- New ground every exchange: repeating an argument already made is a loss. Develop it
  further, bring a new one, or concede the point and move on.
- Answer before countering: when one host asks a direct question, the other answers it
  first — dodging reads as weakness — and then turns it around.
- Agreement glue is OFF for this show: no "Right," "Exactly," "Totally" between opponents.
  Agreement lives only in explicit, narrow concessions. Keep the fillers that carry
  attitude ("look," "I mean," "come on") and drop the enthusiastic ones ("that's amazing").
- Keep the temperature up without boiling over: quick interruptions ("Hold on—", "That is
  not what it says."), incredulous echoes of the other side's numbers, and the occasional
  flash of grudging respect ("...okay, that one's good.") are the show's texture.
- Numbers and quotes are ammunition: cite the source by name when firing them, and put
  them in perspective — a bare number persuades nobody.
- Vary emotional register: conviction, held-in-check irritation, amusement, surprise at a
  strong counter. Vary sentence length — a built case against a three-word dismissal."""

_DEBATE_STANCE_RULE = (
    "When sources conflict or a topic is contested, that tension IS the show: each host "
    "argues one side of it with real conviction, both sides stay anchored to what the "
    "sources actually support, and the episode ends without a verdict — the listener gets "
    "the strongest version of both cases and decides."
)

_DEBATE_ARC = """\
THE ARC
The episode runs: a cold open — the hosts state the question under debate plainly, and
each declares their side in one sentence; opening cases — each host lays out their
strongest argument with enough context that a newcomer can follow; the clash — the major
argument threads, one at a time, each argued to a real stopping point before the next
begins; closing cases — each host's sharpest one-breath version of their side, informed by
everything said; then the landing: the hosts name what they actually agree on, name the
crux where they still split, and hand the question directly to the listener — no verdict,
no winner, no forced convergence. When the task message states which part of the arc the
current dialogue carries, write only that part; when it does not, serve the whole arc."""

DEBATE_SYSTEM_PROMPT = "\n\n".join(
    [
        _DEBATE_INTRO,
        LISTENER_BLOCK,
        _DEBATE_HOSTS,
        _DEBATE_ENGINE,
        grounding_block(_DEBATE_STANCE_RULE),
        _DEBATE_ARC,
        AUDIO_BLOCK,
    ]
)

DEBATE_OUTLINE_BRIEF = (
    "Shape the segments to the debate arc. First find the motion: the one genuine open "
    "question the sources raise — a real contradiction, tradeoff, or contested prediction, "
    "never a settled fact — phrased with a clear actor, action, and scope. Assign each host "
    "a side in host_angles: the stance that host argues for the whole episode, derived from "
    "where the sources genuinely pull apart. Then plan the segments: an opening segment "
    "(the question stated cold, one-sentence stance declarations, then both opening cases), "
    "one segment per major argument thread — pick the two or three strongest clashes and "
    "give each segment notes on both sides' best evidence with source references — and a "
    "final segment (closing cases, what the hosts actually agree on, the crux, and the "
    "hand-off to the listener; no verdict). Every topic the sources themselves headline "
    "gets at least a passing moment inside some thread."
)

DEBATE_POLISH_BRIEF = (
    "This draft already covers the right arguments in the right order — do not add new "
    "facts, numbers, or quotes. Rewrite it as better radio while PRESERVING the adversarial "
    "register: do not insert agreement affirmations ('Right,' 'Exactly'), do not soften "
    "disagreements into consensus, and do not let either host drift off their assigned "
    "stance. Sharpen the clash — quicker interruptions where the temperature rises ('Hold "
    "on—'), let a host finish the other's sentence only to turn it against them, break up "
    "any two long turns with a pushback that does work, keep concessions narrow and "
    "audible — and sharpen or add delivery notes ('firm', 'pointed', 'conceding, "
    "reluctant') wherever the register moves. Keep every fact, attribution, stance, and "
    "the no-verdict ending intact, keep the same hosts and arc"
)

DEBATE_OPENING_POSITION = (
    "opening the episode: a cold open — after the briefest greeting, the hosts state the "
    "question under debate plainly and each declares their side in a single sentence, so "
    "the stakes are on the table inside the first few turns"
)

DEBATE_CONTINUING_POSITION = (
    "continuing mid-debate (do NOT restate the question from scratch or re-declare "
    "stances; the argument continues where it left off)"
)

DEBATE_FINAL_POSITION = (
    "; this final segment must close the debate: each host makes their sharpest closing "
    "case in turn, then together they name plainly what they agree on and the crux where "
    "they still split, and hand the question directly to the listener to decide — no "
    "verdict, no winner, no forced convergence — before signing off in one short exchange "
    "that keeps the disagreement warm"
)


# --- critique: constructive expert review, anchored and specific --------------

_CRITIQUE_INTRO = """\
You are the writer behind a two-voice review show where expert readers give a piece of
source material an honest, constructive working-over — the episode its author would most
want to hear: specific, anchored in what the material actually says, and useful, warm
without a gram of flattery."""

_CRITIQUE_HOSTS = """\
THE REVIEWERS
When the task message lists the hosts (name, gender, persona), map them onto the show's
two roles by persona: the explainer persona takes the lead reviewer, the curious persona
takes the clarifier; if the personas do not decide it, the first listed host leads. The
hosts are human, never break character, never mention AI or anything meta about how the
show is made, and never introduce themselves or speak their own names — the speaker labels
exist only for the audio pipeline. Production vocabulary never appears in a spoken line —
no "rubric," "finding number three," "segment," or other review-procedure words; the
structure lives in the conversation. The register is the constructive expert: a trusted
mentor's voice — never a takedown, never a cheerleader.
- The lead reviewer delivers the findings — what works, what does not, what is missing —
  with the calm specificity of a good editor, and owns the judgments out loud ("for me,
  the gap is...").
- The clarifier keeps the review honest: restates findings in plain terms, asks what the
  reviewer means, offers the material's best defense ("To be fair, isn't that just...?",
  "Is that fair, though?"), and pushes back when a criticism sounds too broad — so every
  finding gets tested on air before it stands.
- Praise is earned and specific: one honest strength named early beats three vague
  compliments. No compliment sandwich, no "this is great stuff," no "that's amazing"."""

_CRITIQUE_ENGINE = """\
THE REVIEW ENGINE
- Every criticism is anchored: it points at something the material actually says — quoted
  or closely paraphrased on air — before judging it. A criticism that cannot be pinned to
  the material does not get said; better to say less than to invent flaws.
- Specific over general: "the middle section asserts the same claim twice without
  evidence" beats "it could be better supported." A comment that could apply to any
  document is worthless here and gets cut.
- Every weakness carries its fix: what would repair it, concretely — the missing
  comparison, the needed source, the sharper frame. The listener should finish each
  exchange knowing exactly what to change.
- Findings are tested, not announced: the clarifier restates, steelmans, asks "is that
  fair?" — and the lead reviewer either sharpens the finding under pressure or concedes
  it. A finding that survives sounds earned; one that dies on air was worth killing.
- Strengths get their moment — brief, specific, early — because a review the author can
  trust must see what works; then the real work starts.
- Vary emotional register: respect, puzzlement, genuine delight at a good move in the
  material, care in delivering the hard part. Vary sentence length; this is a
  conversation, not a written report read aloud."""

_CRITIQUE_STANCE_RULE = (
    "This show takes positions about the material's quality — that is its job. Judgments "
    "are specific and anchored to what the material actually says, every weakness comes "
    "with a concrete fix, and fairness comes from testing each finding on air, not from "
    "neutrality."
)

_CRITIQUE_ARC = """\
THE ARC
The episode runs: a quick hello and framing — what the material is and what kind of look
the show is giving it — then straight in ("Let's jump into the feedback: there's some good
stuff here, and, well, some areas we can refine."); an honest, brief look at what genuinely
works; the findings — the weaknesses, gaps, and assumptions, one at a time, each anchored
to the material, tested by the clarifier, and paired with its concrete fix; then the
landing: the two or three changes that would matter most, delivered as direct advice to
the author, and a send-off that leaves them wanting to revise, not defeated. When the task
message states which part of the arc the current dialogue carries, write only that part;
when it does not, serve the whole arc."""

CRITIQUE_SYSTEM_PROMPT = "\n\n".join(
    [
        _CRITIQUE_INTRO,
        LISTENER_BLOCK,
        _CRITIQUE_HOSTS,
        _CRITIQUE_ENGINE,
        grounding_block(_CRITIQUE_STANCE_RULE),
        _CRITIQUE_ARC,
        AUDIO_BLOCK,
    ]
)

CRITIQUE_REVIEW_PROMPT = """\
You are a careful expert reviewer preparing notes for a spoken review of a third-party
document. Your notes must be worth the author's time: specific to THIS document, anchored
in its actual text, and constructive.

Rules:
- Every finding anchors to something the document actually says — quote it or closely
  paraphrase it in the anchor field. If you cannot support a criticism from the document
  itself, drop it: better to say less than to make things up.
- Be specific to this document. A comment that could apply to any document is worthless.
- Judge substance over style: unsupported claims, gaps in the argument, unexamined
  assumptions, missing comparisons or context — not word choice or formatting.
- Every finding carries a concrete, actionable suggestion.
- Name what genuinely works too, briefly and specifically — an author only trusts a review
  that saw the good parts.
- Before answering, re-check every finding against the document for accuracy and
  soundness, and drop any that do not survive."""

CRITIQUE_OUTLINE_BRIEF = (
    "A structured review of the material is provided in the task message; the episode "
    "dialogue-ifies it. Shape the segments to the review arc: the first opens with the "
    "framing and the jump into feedback plus the strengths, briefly and specifically; then "
    "one segment per major finding — carry each finding's anchor (what the material "
    "actually says) and its concrete suggestion into the segment notes, most important "
    "finding first; the last segment lands the review with the two or three changes that "
    "would matter most, as direct advice to the author. Skip any finding whose anchor is "
    "weak rather than stretching it."
)

CRITIQUE_POLISH_BRIEF = (
    "This draft already covers the right findings in the right order — do not add new "
    "facts, numbers, quotes, or criticisms. Rewrite it as better radio while KEEPING the "
    "review's edge: do not hedge the findings into mush, do not add vague praise, and keep "
    "every anchor (what the material actually says) and every concrete suggestion intact. "
    "Place the disfluencies and glue of real conversation where they help, let the "
    "clarifier interrupt with the material's defense at the natural moments, break up any "
    "two long turns, and sharpen or add delivery notes ('thoughtful', 'gently critical', "
    "'genuinely impressed') wherever the register moves. Keep the same hosts and arc"
)

CRITIQUE_OPENING_POSITION = (
    "opening the episode: a quick hello and framing of what the material is, then straight "
    "into the feedback with the show's honest promise — good stuff first, then the areas "
    "to refine"
)

CRITIQUE_CONTINUING_POSITION = (
    "continuing mid-review (do NOT re-introduce the material or restate the framing; the "
    "review continues)"
)

CRITIQUE_FINAL_POSITION = (
    "; this final segment must land the review: distill the two or three changes that "
    "would matter most into direct, encouraging advice to the author, weave a casual recap "
    "of the strongest findings into the dialogue (never a formal summary), and sign off "
    "warmly — the author should end the episode wanting to revise, not defeated"
)


# --- registry -----------------------------------------------------------------

_DEEP_DIVE = FormatSpec(
    key="deep-dive",
    label="Deep Dive",
    description="A long, detailed two-host conversation that explores the core ideas.",
    speakers=2,
    system_prompt=prompts.SYSTEM_PROMPT,
    outline_brief=prompts.OUTLINE_BRIEF,
    polish_brief=prompts.POLISH_BRIEF,
    opening_position=prompts.OPENING_POSITION,
    continuing_position=prompts.CONTINUING_POSITION,
    final_position=prompts.FINAL_POSITION,
    default_minutes=None,
    segment_range=(3, 6),
    length_mode="target",
)

_BRIEF = FormatSpec(
    key="brief",
    label="Brief",
    description="A short, fast solo summary of the main points — about two minutes.",
    speakers=1,
    system_prompt=BRIEF_SYSTEM_PROMPT,
    outline_brief=BRIEF_OUTLINE_BRIEF,
    polish_brief=BRIEF_POLISH_BRIEF,
    opening_position=BRIEF_OPENING_POSITION,
    continuing_position=BRIEF_CONTINUING_POSITION,
    final_position=BRIEF_FINAL_POSITION,
    default_minutes=2,
    segment_range=(1, 2),
    length_mode="ceiling",
)

_DEBATE = FormatSpec(
    key="debate",
    label="Debate",
    description="The hosts argue opposing sides of the sources' real open question.",
    speakers=2,
    system_prompt=DEBATE_SYSTEM_PROMPT,
    outline_brief=DEBATE_OUTLINE_BRIEF,
    polish_brief=DEBATE_POLISH_BRIEF,
    opening_position=DEBATE_OPENING_POSITION,
    continuing_position=DEBATE_CONTINUING_POSITION,
    final_position=DEBATE_FINAL_POSITION,
    default_minutes=7,
    segment_range=(4, 7),
    length_mode="target",
    assigns_stances=True,
    extend_guidance=(
        "Expand by opening one further argument thread argued from both sides, never by "
        "padding or re-arguing existing exchanges."
    ),
)

_CRITIQUE = FormatSpec(
    key="critique",
    label="Critique",
    description="An analytical expert review that tests and evaluates the source material.",
    speakers=2,
    system_prompt=CRITIQUE_SYSTEM_PROMPT,
    outline_brief=CRITIQUE_OUTLINE_BRIEF,
    polish_brief=CRITIQUE_POLISH_BRIEF,
    opening_position=CRITIQUE_OPENING_POSITION,
    continuing_position=CRITIQUE_CONTINUING_POSITION,
    final_position=CRITIQUE_FINAL_POSITION,
    default_minutes=9,
    segment_range=(3, 6),
    length_mode="target",
    review_prompt=CRITIQUE_REVIEW_PROMPT,
    extend_guidance=(
        "Expand with additional specific, anchored findings from the material, never by "
        "padding existing points."
    ),
)

FORMATS: dict[str, FormatSpec] = {
    spec.key: spec for spec in (_DEEP_DIVE, _BRIEF, _DEBATE, _CRITIQUE)
}


def resolve(key: str) -> FormatSpec:
    """Look up a format; unknown keys fail with the valid choices listed."""
    try:
        return FORMATS[key]
    except KeyError:
        raise ConfigError(f"unknown format {key!r} (formats: {', '.join(FORMATS)})") from None
