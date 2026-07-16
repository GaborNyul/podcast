# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
"""Workspace artifact contract tests (JSON Schema); run in the CI contracts job."""

import json
from pathlib import Path
from typing import cast

import jsonschema
import pytest

from conftest import REPO_ROOT
from podcast import workspace as ws
from podcast.ingest.loader import SourceDocument
from podcast.script.models import Outline, OutlineSegment, Transcript, Turn

pytestmark = pytest.mark.contracts

SCHEMAS_DIR = REPO_ROOT / "schemas"


def _committed(name: str) -> dict[str, object]:
    payload: object = json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast("dict[str, object]", payload)


class TestSchemaStability:
    @pytest.mark.parametrize("name", ["sources", "outline", "transcript"])
    def test_models_still_match_committed_schema(self, name: str) -> None:
        generated = ws.artifact_schemas()[name]
        assert generated == _committed(name), (
            f"{name} artifact schema changed; this is a breaking contract change — "
            f"regenerate schemas/{name}.schema.json deliberately and note it in the PR"
        )


class TestArtifactsValidate:
    def test_sources_artifact(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_sources(workspace, [SourceDocument(path="a.md", title="A", markdown="#", tokens=1)])
        payload = json.loads(workspace.sources_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, _committed("sources"))

    def test_outline_artifact(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_outline(
            workspace, Outline(title="T", segments=[OutlineSegment(heading="h", target_words=5)])
        )
        payload = json.loads(workspace.outline_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, _committed("outline"))

    def test_transcript_artifact(self, tmp_path: Path) -> None:
        workspace = ws.create_workspace(tmp_path, "demo")
        ws.save_transcript(
            workspace,
            Transcript(title="T", hosts=["A", "B"], turns=[Turn(speaker="A", text="x")]),
        )
        payload = json.loads(workspace.transcript_path.read_text(encoding="utf-8"))
        jsonschema.validate(payload, _committed("transcript"))
