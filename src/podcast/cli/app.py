"""Typer application: the single CLI boundary where typed errors become exit codes."""

from typing import Annotated

import typer

from podcast import __version__, doctor
from podcast.cli import ui
from podcast.config import load_config
from podcast.errors import PodcastError

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


def main() -> None:
    """Console-script entry point; renders PodcastError cleanly with its exit code."""
    try:
        app()
    except PodcastError as exc:
        ui.err.print(f"[fail]error:[/] {exc}")
        raise SystemExit(exc.exit_code) from exc
