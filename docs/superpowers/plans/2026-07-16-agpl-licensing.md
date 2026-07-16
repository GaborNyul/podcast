# AGPLv3 Licensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** License the repository under `AGPL-3.0-or-later` and apply it consistently across the `LICENSE` file, packaging metadata, README, every first-party source file, and the CLI.

**Architecture:** Almost entirely declarative — a canonical `LICENSE` file, `pyproject.toml`/`README` edits, and a two-line SPDX header on ~83 first-party `.py` files applied by a throwaway codemod. The only executable additions are DRY license constants in `podcast/__init__.py` surfaced by `podcast --version` and `podcast doctor`, both covered by new tests.

**Tech Stack:** Python 3.13, hatchling (PEP 639), Typer + Rich CLI, pytest + pytest-cov (100% required), pre-commit (ruff, mypy, pyright, pip-licenses, commit-msg gate).

**Spec:** `docs/superpowers/specs/2026-07-16-agpl-licensing-design.md`

**Commit-message constraint (repo gate):** `scripts/check_commit_msg.py` rejects any commit message containing `Co-Authored-By: …Claude`, `Claude-Session:`, `Generated with [Claude Code`, or `noreply@anthropic.com`. **Never add an assistant signature to a commit message in this repo.**

**Environment note:** The worktree has not been `uv sync`-ed yet. The first `uv run …` step below will sync the core environment; allow time for it.

---

### Task 1: Add the verbatim AGPLv3 `LICENSE` file

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Fetch the canonical AGPLv3 text**

Run:
```bash
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o LICENSE
```
If offline, obtain the identical text from a local Python package that bundles it, or from `https://www.gnu.org/licenses/agpl-3.0.txt` via any available mirror. The file must be the **unmodified** FSF text.

- [ ] **Step 2: Verify it is the real, complete license**

Run:
```bash
head -n 1 LICENSE; echo "---"; grep -c "Version 3, 19 November 2007" LICENSE; wc -l < LICENSE
```
Expected: first line `                    GNU AFFERO GENERAL PUBLIC LICENSE`; the grep count is `1`; line count is in the 600–700 range (canonical is 661). If any check fails, re-fetch — do not hand-edit the body.

- [ ] **Step 3: Commit**

```bash
git add LICENSE
git commit -m "chore(license): add verbatim AGPLv3 LICENSE file"
```

---

### Task 2: Declare AGPLv3 in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:11` (the `license = { text = "MIT" }` line, inside `[project]`)

- [ ] **Step 1: Replace the license field and add classifier + URLs**

In `[project]`, replace:
```toml
license = { text = "MIT" }
```
with:
```toml
license = "AGPL-3.0-or-later"
license-files = ["LICENSE"]
classifiers = [
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
]
```
Then add a new top-level table (place it directly after the `[project]` table's scalar fields, before `[project.optional-dependencies]`):
```toml
[project.urls]
Homepage = "https://github.com/gabornyul/podcast"
Repository = "https://github.com/gabornyul/podcast"
```

- [ ] **Step 2: Verify the build backend accepts the SPDX form**

Run:
```bash
uv build 2>&1 | tail -n 5 && rm -rf dist/
```
Expected: build succeeds (`Successfully built …`).
**If it fails** complaining about the `license`/`license-files` keys, the pinned hatchling predates PEP 639 — apply the fallback: set `license = { text = "AGPL-3.0-or-later" }`, delete the `license-files` line, keep the `classifiers`, and re-run `uv build`. `dist/` is git-ignored; the `rm -rf dist/` keeps the tree clean.

- [ ] **Step 3: Verify the pip-licenses self-scan still passes (the AGPL/GPL-3.0 substring hazard)**

Refresh the editable package metadata, then run the exact hook the repo uses:
```bash
uv sync --reinstall-package podcast
uv run pip-licenses --fail-on GPL-3.0 --packages podcast
uv run pip-licenses --fail-on GPL-3.0
```
Expected: both exit `0`. The first line proves our own now-AGPL package does **not** trip the `--fail-on GPL-3.0` guard (exact-match, so `AGPL-3.0-or-later` ≠ `GPL-3.0`).
**If the guard trips** on `podcast`, add `--ignore-packages podcast` to the hook `entry` in `.pre-commit-config.yaml` (the `pip-licenses` hook, ~line 87) **and** append `ignore-packages = ["podcast"]` under `[tool.pip-licenses]`, then re-run.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml
git commit -m "chore(license): declare AGPL-3.0-or-later in package metadata"
```

---

### Task 3: Add a License section to the README

**Files:**
- Modify: `README.md` (append at end of file)

- [ ] **Step 1: Append the License section**

Add this block as the final section of `README.md`:
```markdown
## License

Licensed under the **GNU Affero General Public License v3.0 or later (AGPLv3+)** —
see [LICENSE](LICENSE). In short: any distributed or **network-hosted** derivative must
also be released under the AGPLv3, with source offered to its users. Podcasts you
generate are your own — program output is not a derivative of the program.

Copyright (C) 2026 Gabor Nyul
```

- [ ] **Step 2: Verify**

Run:
```bash
grep -n "GNU Affero General Public License" README.md && grep -n "\[LICENSE\](LICENSE)" README.md
```
Expected: both lines found.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(license): document AGPLv3 in the README"
```

---

### Task 4: DRY license constants + `podcast --version` notice

**Files:**
- Modify: `src/podcast/__init__.py`
- Modify: `src/podcast/cli/app.py:12` (import) and `:54-56` (version branch)
- Test: `tests/test_app.py` (class `TestMainOptions`)

- [ ] **Step 1: Write the failing test**

In `tests/test_app.py`, add to `class TestMainOptions`:
```python
    def test_version_flag_shows_license_and_source(self) -> None:
        result = runner.invoke(app_mod.app, ["--version"])
        assert result.exit_code == 0
        assert "AGPL-3.0-or-later" in result.output
        assert "github.com/gabornyul/podcast" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_app.py::TestMainOptions::test_version_flag_shows_license_and_source -v`
Expected: FAIL (the license/source line is not printed yet).

- [ ] **Step 3: Add the constants**

In `src/podcast/__init__.py`, below `__version__`:
```python
__version__ = "0.1.0"
LICENSE = "AGPL-3.0-or-later"
SOURCE_URL = "https://github.com/gabornyul/podcast"
```

- [ ] **Step 4: Print the notice in the version branch**

In `src/podcast/cli/app.py`, change the import on line 12:
```python
from podcast import LICENSE, SOURCE_URL, __version__, doctor
```
and the version branch (lines 54–56) from:
```python
    if version:
        ui.out.print(f"podcast {__version__}")
        raise typer.Exit()
```
to:
```python
    if version:
        ui.out.print(f"podcast {__version__}")
        ui.out.print(f"{LICENSE} · {SOURCE_URL}")
        raise typer.Exit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestMainOptions -v`
Expected: PASS (both the existing `test_version_flag` and the new test).

- [ ] **Step 6: Commit**

```bash
git add src/podcast/__init__.py src/podcast/cli/app.py tests/test_app.py
git commit -m "feat(cli): show license and source in --version output"
```

---

### Task 5: `podcast doctor` license/source notice

**Files:**
- Modify: `src/podcast/cli/app.py:62-69` (`doctor_command`)
- Test: `tests/test_app.py` (class `TestDoctorCommand`)

- [ ] **Step 1: Write the failing test**

In `tests/test_app.py`, add to `class TestDoctorCommand`:
```python
    @pytest.mark.usefixtures("isolated_env")
    def test_shows_license_and_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "podcast.cli.app.doctor.run_checks",
            _fake_checks([CheckResult(name="ffmpeg", ok=True, detail="version 7.1")]),
        )
        result = runner.invoke(app_mod.app, ["doctor"])
        assert "AGPL-3.0-or-later" in result.output
        assert "github.com/gabornyul/podcast" in result.output
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_app.py::TestDoctorCommand::test_shows_license_and_source -v`
Expected: FAIL (no license footer yet).

- [ ] **Step 3: Append the footer in `doctor_command`**

In `src/podcast/cli/app.py`, change `doctor_command` (lines 62–69) from:
```python
@app.command("doctor")
def doctor_command() -> None:
    """Check that this machine can run the tool (ffmpeg, workspace, engines)."""
    config = load_config()
    results = doctor.run_checks(config)
    ui.out.print(ui.checks_table(results))
    if not all(result.ok for result in results):
        raise typer.Exit(1)
```
to:
```python
@app.command("doctor")
def doctor_command() -> None:
    """Check that this machine can run the tool (ffmpeg, workspace, engines)."""
    config = load_config()
    results = doctor.run_checks(config)
    ui.out.print(ui.checks_table(results))
    ui.out.print(f"License: {LICENSE} · source: {SOURCE_URL}")
    if not all(result.ok for result in results):
        raise typer.Exit(1)
```
(`LICENSE`/`SOURCE_URL` are already imported from Task 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_app.py::TestDoctorCommand -v`
Expected: PASS (all three doctor tests).

- [ ] **Step 5: Commit**

```bash
git add src/podcast/cli/app.py tests/test_app.py
git commit -m "feat(cli): show license and source in doctor output"
```

---

### Task 6: SPDX headers on every first-party `.py` (codemod + enforcing test)

The codemod is a **throwaway** run from the scratchpad — NOT committed (committing it under `scripts/` would force 100% coverage on it, per `[tool.coverage.run] source = ["podcast", "scripts"]`). Only the header diffs and the enforcing test are committed.

**Files:**
- Create (throwaway, scratchpad): `/tmp/claude-1000/-home-gabor-nyul-git-repositories-podcast/9fa8507c-e565-41a0-a1d0-3161a1671930/scratchpad/add_license_headers.py`
- Modify: all `.py` under `src/podcast/`, `tests/`, `scripts/` (~83 files, comment-only)
- Test: `tests/test_license_headers.py` (new)

- [ ] **Step 1: Write the enforcing test**

Create `tests/test_license_headers.py`:
```python
"""Every shipped source file must carry the SPDX AGPL header."""

from pathlib import Path

import podcast

_MARKER = "# SPDX-License-Identifier: AGPL-3.0-or-later"
_PACKAGE_ROOT = Path(podcast.__file__).parent


def test_every_package_file_has_spdx_header() -> None:
    missing = [
        str(path.relative_to(_PACKAGE_ROOT))
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if _MARKER not in path.read_text(encoding="utf-8")
    ]
    assert missing == [], f"missing SPDX header in: {missing}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_license_headers.py -v`
Expected: FAIL — `missing SPDX header in: [...]` listing all package files.

- [ ] **Step 3: Write the throwaway codemod to the scratchpad**

Write this file to `…/scratchpad/add_license_headers.py` (path above):
```python
"""One-shot: insert the SPDX AGPL header into first-party .py files. Idempotent."""

from __future__ import annotations

from pathlib import Path

HEADER = "# SPDX-License-Identifier: AGPL-3.0-or-later\n# Copyright (C) 2026 Gabor Nyul\n"
MARKER = "# SPDX-License-Identifier:"
ROOTS = ("src/podcast", "tests", "scripts")


def with_header(text: str) -> str:
    if MARKER in text:
        return text
    lines = text.splitlines(keepends=True)
    keep = 0
    if lines and lines[0].startswith("#!"):
        keep = 1
    if keep < len(lines) and lines[keep].startswith("#") and "coding:" in lines[keep]:
        keep += 1
    return "".join(lines[:keep]) + HEADER + "".join(lines[keep:])


def main(repo_root: Path) -> None:
    changed = 0
    for root in ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            original = path.read_text(encoding="utf-8")
            updated = with_header(original)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                changed += 1
    print(f"headers added to {changed} files")


if __name__ == "__main__":
    main(Path.cwd())
```

- [ ] **Step 4: Run the codemod from the worktree root**

Run:
```bash
python "$SCRATCH/add_license_headers.py"
```
(where `$SCRATCH` is the scratchpad dir above; run with the worktree root as CWD so `Path.cwd()` is correct).
Expected: `headers added to 84 files` (83 first-party files + the new `test_license_headers.py`).

- [ ] **Step 5: Verify the enforcing test now passes**

Run: `uv run pytest tests/test_license_headers.py -v`
Expected: PASS.

- [ ] **Step 6: Verify the headers broke nothing (lint/format/type + full suite)**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```
Expected: ruff clean, formatting unchanged, full suite green at 100% coverage. If `ruff format --check` reports a file, run `uv run ruff format .` and re-run (headers must not fight the formatter).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore(license): add SPDX AGPL headers to first-party sources"
```
(The scratchpad codemod is outside the repo and is not staged.)

---

### Task 7: Retire TODO #0 and full-repo verification

**Files:**
- Modify: `docs/TODO.md` (remove the `## 0.` section — it has moved to a spec)

- [ ] **Step 1: Remove item #0 from the TODO**

Delete the entire `## 0. Define source code licensing (MIT, GNU, etc.)` section (heading + its paragraph) from `docs/TODO.md`, leaving items 1–3 intact. (Per the TODO's own preamble, "items move into ADRs/specs when they start"; this one is now `docs/superpowers/specs/2026-07-16-agpl-licensing-design.md`.)

- [ ] **Step 2: Full verification**

Run:
```bash
uv run pytest -q
uv run pre-commit run --all-files
```
Expected: full suite green at 100% coverage; every pre-commit hook passes — including `pip-licenses (block GPL-3.0)` and the commit-msg gate wiring.

- [ ] **Step 3: Commit**

```bash
git add docs/TODO.md
git commit -m "docs(todo): retire item #0 — licensing now specced and implemented"
```

---

## Rollout / handoff (not a build task)

After all tasks pass, use `superpowers:finishing-a-development-branch` to push and open the PR. Push to the clean remote branch name:
```bash
git push -u origin HEAD:chore/agpl-license
```
Then open a PR into `main`. (The local branch is `worktree-chore+agpl-license`; the `HEAD:chore/agpl-license` refspec gives the remote the intended name.)

## Self-review notes

- **Spec coverage:** LICENSE (T1) · pyproject field+classifier+urls, DP-3 (T2) · pip-licenses self-scan hazard §4.2/§6, DP-4 (T2 step 3) · README §4.3 (T3) · `--version` §4.5 (T4) · `doctor` §4.5 (T5) · per-file SPDX headers §4.4 + DP-1 short-form + DP-2 file set + enforcing meta-test §7 (T6) · verification §6 + TODO cross-link §9 (T7). All spec sections map to a task.
- **DP-1 (SPDX short-form)** and **DP-2 (src+tests+scripts)** are realized by the Task 6 codemod `HEADER`/`ROOTS`. **DP-4** default (leave `block GPL-3.0` as-is) is honored — the hook is only touched if the self-scan actually trips.
- **Coverage:** the only executable additions are the two constants (import-time, covered) and the two `ui.out.print` lines (covered by T4/T5 tests). Comments, TOML, Markdown, and the LICENSE add no measured lines. Codemod is un-committed, so `scripts/` coverage is unaffected.
- **Type/name consistency:** `LICENSE` and `SOURCE_URL` defined in `podcast/__init__.py` (T4), imported once in `app.py` (T4), reused in `doctor_command` (T5). `_MARKER` string in the enforcing test matches `HEADER` in the codemod exactly.
