"""Rich consoles: data on stdout, diagnostics and progress on stderr."""

from collections.abc import Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

from podcast.doctor import CheckResult

_THEME = Theme({"ok": "bold green", "fail": "bold red", "warn": "yellow", "accent": "cyan"})

out = Console(theme=_THEME)
err = Console(theme=_THEME, stderr=True)


def make_progress() -> Progress:
    """Progress display for multi-step stages (LLM segments, TTS lines)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    )


def checks_table(results: Sequence[CheckResult]) -> Table:
    """Doctor results as a table; hints only shown for failing checks."""
    table = Table(title="podcast doctor", header_style="cyan")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for result in results:
        status = "[bold green]ok[/]" if result.ok else "[bold red]FAIL[/]"
        detail = result.detail if result.ok else f"{result.detail}\n[yellow]{result.hint}[/]"
        table.add_row(result.name, status, detail)
    return table
