"""Typer application: the single CLI boundary where typed errors become exit codes."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress

from podcast import __version__, doctor
from podcast.cli import ui
from podcast.config import AppConfig, load_config
from podcast.errors import PodcastError
from podcast.ingest import load_documents, merged_sources_markdown
from podcast.ingest.tokens import assert_fits_context
from podcast.llm.registry import create_provider
from podcast.script import budget as budget_mod
from podcast.script import pipeline
from podcast.script.models import Transcript
from podcast.workspace import (
    Workspace,
    create_workspace,
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


def _apply_overrides(config: AppConfig, provider_name: str | None, engine: str | None) -> None:
    if provider_name is not None:
        config.llm.provider = provider_name
    if engine is not None:
        config.tts.engine = engine


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

    outline_task = progress.add_task("Planning outline", total=1)
    outline = pipeline.build_outline(provider, config, grounding, budget_words)
    progress.update(outline_task, completed=1)

    dialogue_task = progress.add_task("Writing dialogue", total=len(outline.segments))

    def advance(_index: int) -> None:
        progress.advance(dialogue_task)

    transcript = pipeline.write_dialogue(provider, config, grounding, outline, advance)

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
        int, typer.Option("--duration", "-d", min=0, help="Target minutes (0 = config default).")
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
) -> None:
    """Generate an editable podcast script (script.md) from source documents."""
    config = load_config()
    _apply_overrides(config, provider_name, engine)
    minutes = duration or config.script.default_minutes
    with ui.make_progress() as progress:
        workspace, transcript, _budget = _run_generate(config, sources, minutes, name, progress)
    estimated = budget_mod.estimated_minutes(config, transcript.word_count(), config.tts.engine)
    ui.err.print(
        f"[ok]script ready:[/] {workspace.script_path} "
        f"({transcript.word_count()} words, ~{estimated:.1f} min)"
    )
    ui.err.print("edit the script if you like, then run: [accent]podcast synthesize[/]")
    ui.out.print(str(workspace.root))


def main() -> None:
    """Console-script entry point; renders PodcastError cleanly with its exit code."""
    try:
        app()
    except PodcastError as exc:
        ui.err.print(f"[fail]error:[/] {exc}")
        raise SystemExit(exc.exit_code) from exc
