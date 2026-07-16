# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Tests for podcast.workspace."""

import json
from pathlib import Path

import pytest

from podcast import workspace as ws
from podcast.errors import ScriptError
from podcast.ingest.loader import SourceDocument
from podcast.script.models import Outline, OutlineSegment, Transcript, Turn

TRANSCRIPT = Transcript(
    title="Ants!",
    hosts=["Alex", "Maya"],
    turns=[Turn(speaker="Alex", text="Hello."), Turn(speaker="Maya", text="Hi.")],
)


class TestSlugify:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Ants: A Deep Dive!", "ants-a-deep-dive"),
            ("  spaces  everywhere  ", "spaces-everywhere"),
            ("már ékezetes", "m-r-kezetes"),
            ("!!!", "episode"),
            ("", "episode"),
        ],
    )
    def test_slugs(self, title: str, expected: str) -> None:
        assert ws.slugify(title) == expected


class TestCreateAndOpen:
    def test_create_makes_directories(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path / "episodes", "demo")
        assert workspace.root.is_dir()
        assert workspace.root == tmp_path / "episodes" / "demo"

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ScriptError, match="podcast generate"):
            ws.open_workspace(tmp_path, "ghost")

    def test_open_existing(self, tmp_path: Path) -> None:
        ws.create_workspace(tmp_path, "demo")
        assert ws.open_workspace(tmp_path, "demo").root == tmp_path / "demo"

    @pytest.mark.parametrize("slug", ["../../x", "/abs/path", "a/b", "a\\b", "", ".", ".."])
    def test_create_rejects_slugs_that_escape_episodes_dir(self, tmp_path: Path, slug: str) -> None:
        with pytest.raises(ScriptError, match="simple name"):
            ws.create_workspace(tmp_path / "episodes", slug)
        assert not (tmp_path / "episodes").exists()  # rejected before anything is created
        assert not (tmp_path.parent / "x").exists()

    @pytest.mark.parametrize("slug", ["../../x", "/abs/path", "a/b", "a\\b", "", ".", ".."])
    def test_open_rejects_slugs_that_escape_episodes_dir(self, tmp_path: Path, slug: str) -> None:
        with pytest.raises(ScriptError, match="simple name"):
            ws.open_workspace(tmp_path / "episodes", slug)

    def test_normal_slug_still_works(self, tmp_path: Path) -> None:
        created = ws.create_workspace(tmp_path / "episodes", "normal-slug")
        assert created.root == tmp_path / "episodes" / "normal-slug"
        assert ws.open_workspace(tmp_path / "episodes", "normal-slug").root == created.root

    def test_paths_are_derived_from_root(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        assert workspace.sources_path.name == "sources.json"
        assert workspace.outline_path.name == "outline.json"
        assert workspace.script_path.name == "script.md"
        assert workspace.transcript_path.name == "transcript.json"
        assert workspace.segments_dir.name == "segments"
        assert workspace.episode_path.name == "episode.mp3"


class TestSaveArtifacts:
    def test_save_sources_round_trips(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        documents = [
            SourceDocument(path="a.md", title="A", markdown="# A", tokens=3),
        ]
        ws.save_sources(workspace, documents)
        payload = json.loads(workspace.sources_path.read_text(encoding="utf-8"))
        assert payload[0]["title"] == "A"

    def test_save_outline_writes_json(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        outline = Outline(title="T", segments=[OutlineSegment(heading="h", target_words=5)])
        ws.save_outline(workspace, outline)
        payload = json.loads(workspace.outline_path.read_text(encoding="utf-8"))
        assert payload["segments"][0]["heading"] == "h"

    def test_save_transcript_writes_both_faces(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_transcript(workspace, TRANSCRIPT)
        assert "**Alex:** Hello." in workspace.script_path.read_text(encoding="utf-8")
        payload = json.loads(workspace.transcript_path.read_text(encoding="utf-8"))
        assert payload["title"] == "Ants!"


class TestLoadTranscript:
    def test_reads_script_md(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_transcript(workspace, TRANSCRIPT)
        assert ws.load_transcript(workspace) == TRANSCRIPT

    def test_hand_edit_wins_over_sidecar(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_transcript(workspace, TRANSCRIPT)
        edited = workspace.script_path.read_text(encoding="utf-8").replace(
            "**Maya:** Hi.", "**Maya:** Hi there, edited by hand."
        )
        workspace.script_path.write_text(edited, encoding="utf-8")
        transcript = ws.load_transcript(workspace)
        assert transcript.turns[1].text == "Hi there, edited by hand."

    def test_missing_script_raises(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        with pytest.raises(ScriptError, match="missing"):
            ws.load_transcript(workspace)
