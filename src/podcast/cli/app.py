"""Typer application: the single CLI boundary where typed errors become exit codes."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.progress import Progress
from rich.table import Table

from podcast import __version__, doctor, emphasis
from podcast.audio import pacing
from podcast.audio.assemble import assemble_episode, tempo_variant
from podcast.cli import ui
from podcast.config import AppConfig, load_config
from podcast.errors import PodcastError, ScriptError
from podcast.ingest import load_documents, merged_sources_markdown
from podcast.ingest.tokens import assert_fits_context
from podcast.llm.registry import create_provider
from podcast.script import budget as budget_mod
from podcast.script import formats as formats_mod
from podcast.script import pipeline
from podcast.script.models import Transcript, Turn
from podcast.tts.base import DialogueEngine, DialogueLine
from podcast.tts.cache import CacheStats, ensure_segment
from podcast.tts.registry import available_engines, create_engine
from podcast.tts.voices import resolve_voices, voices_for
from podcast.workspace import (
    Workspace,
    create_workspace,
    load_transcript,
    open_workspace,
    save_outline,
    save_sources,
    save_transcript,
    slugify,
)

app = typer.Typer(
    name="podcast",
    help="Local-first podcast generator: documents in, two-host audio episode out.",
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


@app.callback(invoke_without_command=True)
def main_options(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    """Global options."""
    if version:
        ui.out.print(f"podcast {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        ui.out.print(ctx.get_help())
        raise typer.Exit()


@app.command("doctor")
def doctor_command() -> None:
    """Check that this machine can run the tool (ffmpeg, workspace, engines)."""
    config = load_config()
    results = doctor.run_checks(config)
    ui.out.print(ui.checks_table(results))
    if not all(result.ok for result in results):
        raise typer.Exit(1)


@app.command("config")
def config_command() -> None:
    """Print the fully-resolved configuration as JSON."""
    config = load_config()
    ui.out.print_json(config.model_dump_json())


def _apply_overrides(
    config: AppConfig,
    provider_name: str | None,
    engine: str | None,
    format_key: str | None = None,
) -> None:
    if provider_name is not None:
        config.llm.provider = provider_name
    if engine is not None:
        config.tts.engine = engine
    if format_key is not None:
        # resolve() validates: assignment bypasses the pydantic validator.
        config.script.format = formats_mod.resolve(format_key).key


def _episode_minutes(config: AppConfig, duration: int) -> int:
    """Explicit -d wins; then the format's default; then the config default."""
    if duration:
        return duration
    spec = formats_mod.resolve(config.script.format)
    return spec.default_minutes or config.script.default_minutes


def _run_generate(
    config: AppConfig,
    sources: list[Path],
    minutes: int,
    name: str | None,
    progress: Progress,
) -> tuple[Workspace, Transcript, int]:
    documents = load_documents(sources)
    assert_fits_context(sum(document.tokens for document in documents), config.llm.context_window)
    grounding = merged_sources_markdown(documents)
    provider = create_provider(config)
    budget_words = budget_mod.episode_word_budget(config, minutes, config.tts.engine)

    review = None
    if pipeline.episode_format(config).review_prompt:
        review_task = progress.add_task("Reviewing material", total=1)
        review = pipeline.review_source(provider, config, grounding)
        progress.update(review_task, completed=1)

    outline_task = progress.add_task("Planning outline", total=1)
    outline = pipeline.build_outline(provider, config, grounding, budget_words, review=review)
    progress.update(outline_task, completed=1)

    dialogue_task = progress.add_task("Writing dialogue", total=len(outline.segments))

    def advance(_index: int) -> None:
        progress.advance(dialogue_task)

    transcript = pipeline.write_dialogue(provider, config, grounding, outline, advance)

    if config.script.polish_pass:
        polish_task = progress.add_task("Polishing dialogue", total=1)
        transcript = pipeline.polish_dialogue(provider, config, transcript)
        progress.update(polish_task, completed=1)

    repair_task = progress.add_task("Checking length", total=1)
    transcript = pipeline.ensure_length(provider, config, transcript, budget_words)
    progress.update(repair_task, completed=1)

    workspace = create_workspace(
        config.paths.episodes_dir, name if name is not None else slugify(outline.title)
    )
    save_sources(workspace, documents)
    save_outline(workspace, outline)
    save_transcript(workspace, transcript)
    return workspace, transcript, budget_words


@app.command("generate")
def generate_command(
    sources: Annotated[list[Path], typer.Argument(help="Source documents (txt/md/html/pdf/docx).")],
    duration: Annotated[
        int, typer.Option("--duration", "-d", min=0, help="Target minutes (0 = format default).")
    ] = 0,
    provider_name: Annotated[
        str | None, typer.Option("--provider", help="LLM provider override.")
    ] = None,
    engine: Annotated[
        str | None, typer.Option("--engine", help="TTS engine the length is calibrated for.")
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", help="Episode slug (default: from the title).")
    ] = None,
    format_key: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Audio overview format (see `podcast formats`)."),
    ] = None,
) -> None:
    """Generate an editable podcast script (script.md) from source documents."""
    config = load_config()
    _apply_overrides(config, provider_name, engine, format_key)
    minutes = _episode_minutes(config, duration)
    with ui.make_progress() as progress:
        workspace, transcript, _budget = _run_generate(config, sources, minutes, name, progress)
    estimated = budget_mod.estimated_minutes(config, transcript.word_count(), config.tts.engine)
    ui.err.print(
        f"[ok]script ready:[/] {workspace.script_path} "
        f"({transcript.word_count()} words, ~{estimated:.1f} min)"
    )
    ui.err.print("edit the script if you like, then run: [accent]podcast synthesize[/]")
    ui.out.print(str(workspace.root))


def _latest_slug(episodes_dir: Path) -> str:
    candidates = (
        [entry for entry in episodes_dir.glob("*/script.md") if entry.is_file()]
        if episodes_dir.is_dir()
        else []
    )
    if not candidates:
        raise ScriptError(f"no episodes under {episodes_dir}; run `podcast generate` first")
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.parent.name


def _dialogue_segments(
    engine: DialogueEngine,
    workspace: Workspace,
    spoken: list[Turn],
    voices: dict[str, str],
    composed: Callable[[Turn], str],
    spoken_text: Callable[[Turn], str],
    stats: CacheStats,
    progress: Progress,
) -> list[Path]:
    """Whole-conversation render: any line change re-renders the dialogue, since
    every line's prosody depends on the lines before it."""
    lines = [
        DialogueLine(speaker=turn.speaker, text=spoken_text(turn), delivery=composed(turn))
        for turn in spoken
    ]
    digest = hashlib.sha256()
    for voice in sorted({voices[line.speaker] for line in lines}):
        for part in (voice, engine.cache_token(voice)):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
    for line in lines:
        # line.speaker joins the key: dialogue engines derive their speaker-slot
        # assignment (SoulX's [S1]/[S2]) from the speaker sequence, so two hosts
        # sharing one resolved voice still produce different audio when lines
        # swap speakers — the voice alone cannot distinguish those renders.
        for part in (engine.name, line.speaker, voices[line.speaker], line.text, line.delivery):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
    key = digest.hexdigest()[:32]
    out_paths = [
        workspace.segments_dir / f"dialogue-{key}-{index:04d}.wav" for index in range(len(lines))
    ]
    task = progress.add_task("Synthesizing dialogue", total=1)
    if all(path.is_file() for path in out_paths):
        stats.hits += len(lines)
    else:
        workspace.segments_dir.mkdir(parents=True, exist_ok=True)
        engine.synthesize_dialogue(lines, voices, out_paths)
        stats.misses += len(lines)
    progress.update(task, completed=1)
    return out_paths


def _run_synthesize(config: AppConfig, workspace: Workspace, progress: Progress) -> CacheStats:
    transcript = load_transcript(workspace)
    engine = create_engine(config)
    info = engine.info()
    voices = resolve_voices(config, engine.name, transcript.hosts)
    stats = CacheStats()
    spoken = [turn for turn in transcript.turns if turn.text.strip()]
    supports_delivery = info.supports_delivery
    supports_emphasis = info.supports_emphasis
    styles = {host.name: host.style for host in config.script.hosts}
    tempos = {host.name: host.tempo for host in config.script.hosts}

    def composed(turn: Turn) -> str:
        if not supports_delivery:
            return ""
        return "; ".join(part for part in (styles.get(turn.speaker, ""), turn.delivery) if part)

    def spoken_text(turn: Turn) -> str:
        if not supports_emphasis:
            return emphasis.strip_markup(turn.text)
        return turn.text

    rendered_paths: list[Path] = []
    if info.dialogue_native and isinstance(engine, DialogueEngine):
        rendered_paths = _dialogue_segments(
            engine, workspace, spoken, voices, composed, spoken_text, stats, progress
        )
    else:
        task = progress.add_task("Synthesizing lines", total=len(spoken))
        for turn in spoken:
            voice = voices[turn.speaker]
            delivery = composed(turn)
            line_text = spoken_text(turn)

            def render(
                path: Path, text: str = line_text, voice_id: str = voice, note: str = delivery
            ) -> None:
                engine.synthesize_line(text, voice_id, path, delivery=note)

            rendered_paths.append(
                ensure_segment(
                    workspace.segments_dir, engine.name, voice, line_text, delivery, render, stats
                )
            )
            progress.advance(task)
    segment_paths = [
        tempo_variant(path, tempos.get(turn.speaker, 1.0))
        for path, turn in zip(rendered_paths, spoken, strict=True)
    ]

    assemble_task = progress.add_task("Assembling episode", total=1)
    assemble_episode(
        segment_paths,
        workspace.episode_path,
        work_dir=workspace.segments_dir / "work",
        sample_rate=info.sample_rate,
        pause_min_ms=config.audio.pause_min_ms,
        pause_max_ms=config.audio.pause_max_ms,
        bitrate=config.audio.mp3_bitrate,
        seed=config.audio.seed,
        gap_scales=pacing.gap_scales(spoken),
    )
    progress.update(assemble_task, completed=1)
    return stats


def _report_synthesis(workspace: Workspace, stats: CacheStats) -> None:
    ui.err.print(
        f"[ok]episode ready:[/] {workspace.episode_path} "
        f"({stats.total} segments: {stats.hits} cached, {stats.misses} rendered)"
    )
    ui.out.print(str(workspace.episode_path))


@app.command("synthesize")
def synthesize_command(
    name: Annotated[
        str | None, typer.Argument(help="Episode slug (default: most recent episode).")
    ] = None,
    engine: Annotated[str | None, typer.Option("--engine", help="TTS engine override.")] = None,
) -> None:
    """Render script.md to episode.mp3 (cached: edited lines re-render alone)."""
    config = load_config()
    _apply_overrides(config, None, engine)
    slug = name if name is not None else _latest_slug(config.paths.episodes_dir)
    workspace = open_workspace(config.paths.episodes_dir, slug)
    with ui.make_progress() as progress:
        stats = _run_synthesize(config, workspace, progress)
    _report_synthesis(workspace, stats)


@app.command("create")
def create_command(
    sources: Annotated[list[Path], typer.Argument(help="Source documents (txt/md/html/pdf/docx).")],
    duration: Annotated[
        int, typer.Option("--duration", "-d", min=0, help="Target minutes (0 = format default).")
    ] = 0,
    provider_name: Annotated[
        str | None, typer.Option("--provider", help="LLM provider override.")
    ] = None,
    engine: Annotated[str | None, typer.Option("--engine", help="TTS engine override.")] = None,
    name: Annotated[
        str | None, typer.Option("--name", help="Episode slug (default: from the title).")
    ] = None,
    format_key: Annotated[
        str | None,
        typer.Option("--format", "-f", help="Audio overview format (see `podcast formats`)."),
    ] = None,
) -> None:
    """Generate a script and synthesize the episode in one run."""
    config = load_config()
    _apply_overrides(config, provider_name, engine, format_key)
    minutes = _episode_minutes(config, duration)
    with ui.make_progress() as progress:
        workspace, _transcript, _budget = _run_generate(config, sources, minutes, name, progress)
        stats = _run_synthesize(config, workspace, progress)
    _report_synthesis(workspace, stats)


@app.command("formats")
def formats_command() -> None:
    """List the audio overview formats (brief, deep-dive, debate, critique)."""
    config = load_config()
    table = Table(title="Audio overview formats", header_style="cyan")
    table.add_column("format")
    table.add_column("speakers")
    table.add_column("length")
    table.add_column("description", overflow="fold")
    for spec in formats_mod.FORMATS.values():
        current = " (selected)" if spec.key == config.script.format else ""
        minutes = spec.default_minutes or config.script.default_minutes
        speakers_label = {None: "all hosts", 1: "solo"}.get(spec.speakers, "two hosts")
        table.add_row(
            f"{spec.key}{current}",
            speakers_label,
            f"~{minutes} min",
            spec.description,
        )
    ui.out.print(table)


@app.command("engines")
def engines_command() -> None:
    """List TTS engines and whether this machine can run them."""
    config = load_config()
    table = Table(title="TTS engines", header_style="cyan")
    table.add_column("engine")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for engine_name in available_engines():
        probe = config.model_copy(deep=True)
        probe.tts.engine = engine_name
        result = doctor.check_engine(probe)
        marker = "[bold green]ok[/]" if result.ok else "[bold red]unavailable[/]"
        detail = result.detail if result.ok else f"{result.detail} — {result.hint}"
        current = " (selected)" if engine_name == config.tts.engine else ""
        table.add_row(f"{engine_name}{current}", marker, detail)
    ui.out.print(table)


@app.command("voices")
def voices_command(
    engine: Annotated[
        str | None, typer.Option("--engine", help="Engine to list voices for.")
    ] = None,
) -> None:
    """List an engine's voices and the current speaker mapping."""
    config = load_config()
    _apply_overrides(config, None, engine)
    engine_name = config.tts.engine
    table = Table(title=f"{engine_name} voices", header_style="cyan")
    table.add_column("voice")
    table.add_column("gender")
    for voice in voices_for(engine_name):
        table.add_row(voice.id, voice.gender)
    ui.out.print(table)
    hosts = [host.name for host in config.script.hosts]
    mapping = resolve_voices(config, engine_name, hosts)
    for speaker, voice_id in mapping.items():
        ui.out.print(f"{speaker} -> {voice_id}")


def main() -> None:
    """Console-script entry point; renders PodcastError cleanly with its exit code."""
    try:
        app()
    except PodcastError as exc:
        # Error text may quote script fragments ('[/]', '[laughs]'…): escape it
        # so rich prints it literally; only the [fail] style tag stays markup.
        ui.err.print(f"[fail]error:[/] {escape(str(exc))}")
        raise SystemExit(exc.exit_code) from exc
