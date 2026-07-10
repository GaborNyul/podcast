"""Tests for podcast.cli.ui."""

from rich.console import Console

from podcast.cli import ui
from podcast.doctor import CheckResult


def _render(console_width: int = 100) -> Console:
    return Console(record=True, width=console_width)


class TestConsoles:
    def test_out_writes_to_stdout(self) -> None:
        assert ui.out.stderr is False

    def test_err_writes_to_stderr(self) -> None:
        assert ui.err.stderr is True


class TestMakeProgress:
    def test_progress_renders_to_stderr_console(self) -> None:
        progress = ui.make_progress()
        assert progress.console is ui.err

    def test_progress_has_spinner_bar_and_counts(self) -> None:
        progress = ui.make_progress()
        assert len(progress.columns) == 5


class TestChecksTable:
    def test_shows_ok_rows_without_hint(self) -> None:
        console = _render()
        console.print(ui.checks_table([CheckResult(name="ffmpeg", ok=True, detail="version 7.1")]))
        text = console.export_text()
        assert "ffmpeg" in text
        assert "ok" in text
        assert "FAIL" not in text

    def test_shows_hint_for_failing_rows(self) -> None:
        console = _render()
        console.print(
            ui.checks_table(
                [
                    CheckResult(
                        name="ffmpeg",
                        ok=False,
                        detail="not found on PATH",
                        hint="install ffmpeg",
                    )
                ]
            )
        )
        text = console.export_text()
        assert "FAIL" in text
        assert "install ffmpeg" in text

    def test_empty_results_render(self) -> None:
        console = _render()
        console.print(ui.checks_table([]))
        assert "podcast doctor" in console.export_text()
