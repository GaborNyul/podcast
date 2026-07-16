# Design: Adopt AGPLv3 as the project license (TODO #0)

- **Date:** 2026-07-16
- **Branch:** `chore/agpl-license` (worktree off `origin/main`)
- **Author:** Gabor Nyul
- **Status:** Draft — awaiting user review before writing the implementation plan
- **Source TODO:** `docs/TODO.md` item **#0 — Define source code licensing (MIT, GNU, etc.)**

## 1. Context & problem

The repository is public (`github.com/gabornyul/podcast`) but ships **no `LICENSE`
file**, which legally means *all rights reserved* — nobody may use, modify, or
redistribute it. Meanwhile `pyproject.toml:11` already declares
`license = { text = "MIT" }`, so the packaging metadata and the repository's actual
legal state contradict each other. This task makes a deliberate license choice and
applies it consistently everywhere it must appear.

Constraints that shaped the decision:

- **Sole copyright holder.** `pyproject.toml` lists a single author (Gabor Nyul); there
  are no other contributors, so the license can be chosen freely with no CLA or
  relicensing-consent complications.
- **Dependencies do not force the choice.** Core deps (typer, rich, pydantic, httpx,
  anthropic, kokoro-onnx, …) are permissive MIT/BSD/Apache; `trafilatura>=2.0` is
  Apache-2.0 (it relicensed away from GPL). The qwen3/SoulX **model weights and the
  SoulX inference source are fetched at runtime, not vendored**, so their licenses do
  not propagate into this repository's code. Permissive dependencies are compatible
  with distributing the combined work under AGPL.
- **Only bundled non-code assets** are `assets/voices/soulx/{alex,maya}.{wav,txt}`,
  currently *self-generated* (qwen3-minted), so authorship is clean today. (See §9.)

## 2. Decision

License the repository under the **GNU Affero General Public License, version 3 or
later** — SPDX `AGPL-3.0-or-later`. **Single license. No dual-licensing, no commercial
carve-out.**

**Rationale.** The author's explicit goals are (a) no derivative may ever be
closed-source, and (b) limit others' ability to build a proprietary business on the
code. AGPL is the only mainstream license that also closes the **network/SaaS
loophole** that plain GPL leaves open: anyone who runs a modified version as a hosted
service must offer users the corresponding source. Copyleft does not — and by the Open
Source Definition *cannot* — ban commercial use outright; instead it removes the ability
to build a **closed** product/service on the code, which is the author's stated intent.
A pure single-license approach (not dual-licensing) was chosen deliberately: the author
does not want to reserve a private commercial path.

`-or-later` (vs `-only`) follows the FSF's recommended default so the project can adopt
a future AGPLv4 at its option.

## 3. Scope

**In scope (full hardening — five surfaces):**

1. `LICENSE` file (new) — verbatim AGPLv3 text.
2. `pyproject.toml` — license field, `license-files`, trove classifier (+ optional
   `[project.urls]`).
3. `README.md` — new `## License` section.
4. Per-file license headers across all first-party `.py` (~83 files).
5. CLI license/source notice in `podcast --version` and `podcast doctor`.

**Out of scope (non-goals):**

- Dual-licensing, a commercial license, or any non-commercial/BSL restriction.
- Relicensing or auditing the runtime-downloaded model weights (qwen3, SoulX, Kokoro).
- Deciding rights for *future* NotebookLM-cloned voice assets — that is **TODO #3**
  (§9 cross-links it).
- REUSE-spec `.reuse/dep5` compliance tooling (YAGNI; SPDX headers + `LICENSE` suffice).

## 4. Detailed design

### 4.1 `LICENSE` (new, repo root)

Create `LICENSE` containing the **verbatim** text of the GNU Affero General Public
License v3, fetched from `https://www.gnu.org/licenses/agpl-3.0.txt`. The full text is
mandatory — the license is only effective when reproduced in full. No edits, no
placeholder substitution inside the license body itself.

### 4.2 `pyproject.toml`

Replace the MIT declaration at line 11 and add a classifier. **Primary form** (PEP 639,
SPDX expression — supported by hatchling ≥ 1.27):

```toml
license = "AGPL-3.0-or-later"
license-files = ["LICENSE"]
classifiers = [
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
]
```

**Fallback** (only if the pinned hatchling predates PEP 639 support and `uv build`
fails): keep the table form `license = { text = "AGPL-3.0-or-later" }` plus the
classifier. The build **must** still succeed — verified in §6.

**Optional (recommended while editing this file):** add source-discoverability URLs,
which reinforce AGPL's "where to get the source" intent:

```toml
[project.urls]
Homepage = "https://github.com/gabornyul/podcast"
Repository = "https://github.com/gabornyul/podcast"
```

**Self-scan hazard (must verify).** The pre-commit hook `pip-licenses (block GPL-3.0)`
runs `uv run pip-licenses --fail-on GPL-3.0` (mirrored by `[tool.pip-licenses]
fail-on = "GPL-3.0"` at line ~207). `pip-licenses` scans the whole installed
environment, which **includes the editable `podcast` package itself**. Because the
literal string `AGPL-3.0` contains `GPL-3.0` as a substring, flipping our own license
to AGPL could trip this guard on the licensing commit — depending on whether
`pip-licenses` matches exactly (safe: `AGPL-3.0-or-later` ≠ `GPL-3.0`) or partially
(unsafe). The hook does **not** pass `--partial-match`, so exact matching is expected
and the change should pass — but this must be confirmed by actually running the hook
(§6). If it does trip, the minimal fix is to exclude the self-package
(`pip-licenses --ignore-packages podcast --fail-on GPL-3.0`) in both the hook entry and
`[tool.pip-licenses]`. See also **DP-4** (§8).

### 4.3 `README.md`

Add a `## License` section near the end (README currently has none):

- Names the license: *GNU Affero General Public License v3.0 or later (AGPLv3+)*.
- Links the `LICENSE` file.
- One sentence on the practical implication: **derivatives and network/hosted versions
  must also be released under AGPLv3** (source must be offered to remote users).
- Copyright line: `Copyright (C) 2026 Gabor Nyul`.
- One clarifying sentence that **program output is not a derivative of the program** —
  podcasts a user generates are the user's own and are not encumbered by AGPL.

### 4.4 Per-file license headers

Add a short header to **every first-party `.py` file**: `src/podcast/**` (38),
`tests/**` (39), and `scripts/**` (6) — ~83 files. Third-party/vendored code (none is
committed today) would be excluded.

**Header style — SPDX short-form (chosen; see Decision Point DP-1 in §8):**

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Gabor Nyul
```

Rules:

- Insert at the very top of the file, **except** after a shebang (`#!...`) or an
  existing encoding line, which must remain first (relevant for `scripts/**`).
- Empty `__init__.py` files still receive the header, for uniformity.
- Idempotent: skip any file that already carries an
  `SPDX-License-Identifier` line (re-runnable without duplicating).
- Applied via a small, reviewable script or codemod, not hand-editing 83 files; the
  script is a build/dev utility and is not required to ship.

### 4.5 CLI license/source notice

Extend the two existing surfaces (both already implemented — this is an extension, not
new plumbing):

- **`podcast --version`** — `cli/app.py:48-55` currently prints `podcast {__version__}`.
  Add a second line so the license and source are discoverable:

  ```
  podcast 0.1.0
  AGPL-3.0-or-later · https://github.com/gabornyul/podcast
  ```

- **`podcast doctor`** — `doctor.py` exposes `run_checks() -> list[CheckResult]`, rendered
  by the doctor command in `cli/app.py`. Append a single informational
  license/source line to the doctor output (a footer print, or a dedicated
  informational `CheckResult`), e.g.:

  ```
  License: AGPL-3.0-or-later — source: https://github.com/gabornyul/podcast
  ```

The repository URL (`https://github.com/gabornyul/podcast`) is taken from the README
clone instructions. Prefer sourcing version/URL from one place (`__version__` and, if
`[project.urls]` is added, package metadata) rather than hard-coding strings twice.

## 5. Components & data flow

This change is almost entirely declarative — static files and two small print-path
additions. No new modules, no new runtime data flow. The only executable behavior added
is two extra output lines (version, doctor), both pure functions of constants/metadata.

## 6. Verification

- `uv build` (or equivalent metadata resolution) succeeds with the new license field —
  confirms the PEP 639 form is accepted; if not, apply the §4.2 fallback and re-verify.
- **Run the `pip-licenses (block GPL-3.0)` pre-commit hook against the actual licensing
  commit** and confirm it passes with `podcast` now AGPL (the self-scan hazard in §4.2).
  Apply the `--ignore-packages podcast` fallback only if it trips. Separately confirm no
  `GPLv2-only`/proprietary dependency would conflict with AGPLv3 (permissive
  MIT/BSD/Apache deps are fine).
- `podcast --version` and `podcast doctor` show the new notice (observed by running
  them, not only asserted in tests).
- `LICENSE` byte-matches the canonical FSF AGPLv3 text (length/hash check).
- Full test suite green and the project's **100% coverage** bar maintained (§7).

## 7. Testing

- **`tests/test_app.py`** — extend the `--version` test to assert the license/source
  line is present in output.
- **`tests/test_doctor.py`** — assert the doctor output contains the license/source
  notice.
- **Header presence (optional but recommended):** a lightweight meta-test that every
  first-party `.py` under `src/podcast/` contains an `SPDX-License-Identifier` line —
  cheap insurance that the header pass stays complete as files are added.
- Header/`LICENSE`/README/`pyproject` edits are otherwise static; the coverage
  requirement is driven by the two CLI-output additions, which must be covered.

## 8. Decision points for review

- **DP-1 — header style.** The spec chooses the **SPDX short-form** two-line header
  (modern, machine-verifiable, legally sufficient alongside the full `LICENSE`). The
  alternative is the **full FSF boilerplate** (~14 lines/file: name, copyright,
  warranty disclaimer, "you should have received a copy…"). Full boilerplate is
  marginally more robust but adds ~1,100 lines of comment noise across 83 files. Flip
  to full boilerplate here if preferred.
- **DP-2 — header file set.** Chosen: `src/podcast/**` + `tests/**` + `scripts/**`.
  Narrow to shipped package only (`src/podcast/**`) if headers on tests/scripts feel
  like noise.
- **DP-3 — `[project.urls]`.** Optional addition (§4.2). Include for source
  discoverability, or skip to keep the diff minimal.
- **DP-4 — the `block GPL-3.0` dependency policy.** That hook was written to keep strong
  copyleft *out* of a then-MIT project's dependency tree. Now that the project itself is
  AGPL, GPL/AGPL dependencies would be license-compatible to combine, so the policy is
  arguably obsolete. **Default for this task: leave the policy as-is** (only add
  `--ignore-packages podcast` if the self-scan trips), and treat relaxing/retiring the
  dependency-license gate as separate follow-up work, not part of TODO #0.

## 9. Related work

**TODO #3 (Clone the original NotebookLM voices)** will replace the currently
self-generated `assets/voices/soulx/*.wav` references with clones of *real NotebookLM
host audio*. That raises a distinct IP/ToS question about the **audio assets**, which is
separate from this repository's **source-code** license and is explicitly deferred to
TODO #3 (which already lists the licensing/ToS check as its step 4). Under the AGPL
adopted here, the committed assets ride under the repository license as-is *today*; the
asset-rights question is revisited when those files are swapped.

## 10. Rollout

Single commit set on `chore/agpl-license` (pushed to remote `chore/agpl-license`),
then a PR to `main`. No migration, no runtime/config changes for existing users beyond
the two extra informational output lines.
