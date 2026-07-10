"""Episode workspace: every artifact of one episode lives under episodes/<slug>/."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from podcast.errors import ScriptError
from podcast.ingest.loader import SourceDocument
from podcast.script.markdown import markdown_to_transcript, transcript_to_markdown
from podcast.script.models import Outline, Transcript

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Workspace:
    """Paths of one episode's on-disk artifacts."""

    root: Path

    @property
    def sources_path(self) -> Path:
        return self.root / "sources.json"

    @property
    def outline_path(self) -> Path:
        return self.root / "outline.json"

    @property
    def script_path(self) -> Path:
        return self.root / "script.md"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def episode_path(self) -> Path:
        return self.root / "episode.mp3"


def artifact_schemas() -> dict[str, dict[str, object]]:
    """JSON Schemas of the on-disk workspace artifacts (the public contract)."""
    sources_schema: dict[str, object] = {
        "type": "array",
        "items": SourceDocument.model_json_schema(),
    }
    return {
        "sources": sources_schema,
        "outline": Outline.model_json_schema(),
        "transcript": Transcript.model_json_schema(),
    }


def slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "episode"


def create_workspace(episodes_dir: Path, slug: str) -> Workspace:
    root = episodes_dir / slug
    root.mkdir(parents=True, exist_ok=True)
    return Workspace(root)


def open_workspace(episodes_dir: Path, slug: str) -> Workspace:
    root = episodes_dir / slug
    if not root.is_dir():
        raise ScriptError(f"no episode workspace at {root}; run `podcast generate` first")
    return Workspace(root)


def save_sources(workspace: Workspace, documents: list[SourceDocument]) -> None:
    payload = [document.model_dump() for document in documents]
    workspace.sources_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_outline(workspace: Workspace, outline: Outline) -> None:
    workspace.outline_path.write_text(outline.model_dump_json(indent=2) + "\n", encoding="utf-8")


def save_transcript(workspace: Workspace, transcript: Transcript) -> None:
    """Write both faces of the script contract: editable .md and sidecar .json."""
    workspace.script_path.write_text(transcript_to_markdown(transcript), encoding="utf-8")
    workspace.transcript_path.write_text(
        transcript.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def load_transcript(workspace: Workspace) -> Transcript:
    """Read the script for synthesis; script.md wins because the user may edit it."""
    if not workspace.script_path.is_file():
        raise ScriptError(f"missing {workspace.script_path}; run `podcast generate` first")
    return markdown_to_transcript(workspace.script_path.read_text(encoding="utf-8"))
