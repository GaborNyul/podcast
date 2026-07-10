# Python Development Standards v3

Create by Gabor Nyul
06-Jun-2026

## Python Language

- Python 3.14+ preferred; 3.13 acceptable for projects with dependency constraints
- Type hints everywhere -- no `Any`, no `from __future__ import annotations`
- PEP 8 style compliance enforced
- PEP 604 unions (`X | Y`), PEP 695 type aliases, dataclasses over plain dicts, pathlib over os.path, f-strings
- t-strings (PEP 750): available in 3.14+ only -- do not use in projects targeting 3.13

## Type Checking

- Dual type checking: mypy + pyright, both in strict mode
- mypy configuration in `pyproject.toml`:
  ```toml
  [tool.mypy]
  python_version = "3.14"
  strict = true
  ```
- pyright configuration in `pyproject.toml`:
  ```toml
  [tool.pyright]
  pythonVersion = "3.14"
  typeCheckingMode = "strict"
  ```
- Both must pass in CI and pre-commit before merge

## Project / Venv / Dependencies

- Project management: uv (`pyproject.toml`)
- Lock file: `uv.lock` committed to repo
- Supply chain security: `uv lock --require-hashes` for hash-pinned lockfiles
- Lock file integrity check in CI: `uv lock --check`

## Linting + Formatting

- Tool: ruff (configured in `pyproject.toml`)
- Line length: 100
- Minimum rule sets:
  ```toml
  [tool.ruff]
  line-length = 100
  target-version = "py314"

  [tool.ruff.lint]
  select = [
      "E",    # pycodestyle errors
      "W",    # pycodestyle warnings
      "F",    # pyflakes
      "I",    # isort
      "UP",   # pyupgrade
      "B",    # flake8-bugbear
      "SIM",  # flake8-simplify
      "C4",   # flake8-comprehensions
      "DTZ",  # flake8-datetimez
      "PIE",  # flake8-pie
      "PT",   # flake8-pytest-style
      "RET",  # flake8-return
      "ARG",  # flake8-unused-arguments
      "S",    # flake8-bandit (security)
  ]

  [tool.ruff.lint.per-file-ignores]
  "tests/**" = ["S101", "S603", "S607"]  # allow assert, subprocess in tests
  ```
- Use `ruff S` (flake8-bandit) as the fast security linter -- replaces standalone bandit for most cases

## Testing

### Frameworks and Tools

| Tool | Purpose | Required? |
|------|---------|-----------|
| `pytest` | Test runner | Yes |
| `pytest-cov` | Coverage measurement | Yes |
| `pytest-asyncio` | Async test support (TUI, MCP, FastAPI) | When async code exists |
| `hypothesis` | Property-based / fuzz testing | Yes |
| `jsonschema` | API/CLI contract validation | When JSON output exists |
| `mutmut` | Mutation testing (test quality verification) | Recommended (CI nightly) |
| `behave` | BDD / Gherkin feature tests | Optional |

### Coverage Targets

- Core libraries / CLI: 100% (enforced via `fail_under = 100`)
- Web API / backend: >= 90%
- Frontend: 100% statements + branches (vitest/jest)
- New modules: must not lower overall coverage

### Test Conventions

- Use `tmp_path` (pytest built-in) for filesystem isolation -- never `tempfile` directly
- Use `monkeypatch` for environment/chdir/attribute patching -- never `unittest.mock.patch` as decorator
- Type-annotate all test functions: `def test_foo(self, tmp_path: Path) -> None`
- One test class per public function: `class TestAddDep`, `class TestRemoveDep`
- Test categories per function: happy path, empty/missing data, invalid input, error states, exception cases, edge cases (Unicode, long strings, boundary values)
- Fixture naming: descriptive, indicates what state is created (e.g., `tickets_with_deps`, `config_with_custom_statuses`)

### Property-Based Testing (Hypothesis)

Use hypothesis for:
- Round-trip / serialization invariants (parse -> serialize -> parse)
- Graph algorithm correctness (cycle detection, tree traversal)
- Input validation (search with arbitrary strings never crashes)
- Data integrity (cascade operations clean all references)

Do not use hypothesis for:
- Simple CRUD operations where explicit test cases suffice
- UI rendering tests
- Tests requiring specific, meaningful fixtures

### Mutation Testing

- Tool: `mutmut`
- Run per-module after implementation: `mutmut run --paths-to-mutate src/module.py`
- Quality gate: <5% surviving mutants per module; <3% for critical modules
- CI: nightly scheduled job on changed files (not PR-blocking)

### Performance Testing

- Mark with `@pytest.mark.performance`
- Exclude from default pytest run
- Run nightly in CI: `pytest -m performance --timeout=60`
- Define explicit timing assertions: `assert elapsed < 2.0`
- Test with realistic dataset sizes (1000+ items)

## Security Scanning

### Layer 1: Fast (in-editor / pre-commit)
- ruff `S` rules (flake8-bandit) -- catches hardcoded passwords, insecure temp files, subprocess injection
- `detect-secrets` -- prevents accidentally committed API keys, tokens, passwords

### Layer 2: Pre-commit hooks
- `bandit -r -ll` -- deeper Python-specific security analysis
- Note: ruff `S` rules overlap with bandit. If ruff `S` coverage is sufficient for your project, bandit can be dropped from pre-commit to reduce hook runtime. Keep bandit in CI regardless.

### Layer 3: CI pipeline
- SAST: `semgrep --config p/python --config p/owasp-top-ten --config p/secrets --error`
- Dependency audit: `pip-audit` (checks PyPI advisories)
- License compliance: `pip-licenses --fail-on GPL-3.0` (block copyleft if needed)
- Supply chain: `uv lock --check` (verify lockfile integrity)

### Layer 4: Container / Infrastructure
- Container scanning: `trivy image <image> --severity HIGH,CRITICAL --exit-code 1`
- Run after every container image rebuild
- Block deployment on HIGH/CRITICAL findings

## Automated Adversarial Code Review

Automated adversarial review assumes a reviewer will *always* find something. The standard therefore does not aim for "zero findings" -- it aims for a **bounded, auditable** review-fix loop that converges on the findings that matter and explicitly flags the rest.

### Roles (reviewer != fixer)

- **Reviewer agent**: prompted adversarially -- its job is to actively try to break the code and surface faults, not to confirm that it works.
- **Fixer agent**: applies fixes for findings the loop is responsible for.
- Reviewer and fixer MUST be separate roles/instances. The author (human or agent) does not grade its own homework.

### Execution model

- Runs **agent-driven, pre-PR** on the local working branch, before a PR is opened.
- The existing CI gates (tests, type checking, security scanning) remain the downstream validation. The adversarial loop is an additional pre-PR quality stage, not a replacement for CI.
- Invoked via an OS-agnostic Python quality-gate script (e.g. `scripts/adversarial_review.py`) consistent with the Quality Gate Scripts section.

### Severity model (4-tier)

| Severity | Loop behaviour |
|----------|----------------|
| `Critical` | Must-fix -- loop keeps iterating |
| `High` | Must-fix -- loop keeps iterating |
| `Medium` | Residual -- never blocks the loop; logged and carried into the report |
| `Low` | Residual -- never blocks the loop; logged and carried into the report |

- Severity is assigned by the reviewer and **recorded** in the audit trail. A finding may not be silenced by re-grading it downward (anti-gaming).

### Loop termination -- three gates, whichever trips first

1. **Convergence (primary gate)**: stop when a round produces **no new Critical/High findings**. Outcome = pass.
2. **Hard iteration cap (backstop)**: **configurable in `pyproject.toml`, default 5 rounds**. If the cap is hit with Critical/High findings still open, **halt and flag for the user** -- do not auto-accept.
3. **Oscillation detector**: if the same finding resurfaces, or a fix-A-breaks-B-then-fix-B-breaks-A pattern is detected, **halt immediately** and escalate to the user as a *design-level conflict* requiring human judgment. Do not waste rounds thrashing.

Additional per-round backstop: every fix round MUST re-pass the existing quality gates (ruff, mypy, pyright, pytest). A fix that breaks any gate counts as **no-progress** for that round and does not advance convergence.

### Outcome rules (what "done" means)

- **Auto-accept (pass)** -- permitted *only* when all remaining findings are sub-threshold (Medium/Low). Residuals are logged, not fixed. No human gate required.
- **Flag for sign-off** -- any Critical/High finding still open at stop (cap reached or oscillation halt) requires explicit human review before merge.
- The loop **never silently passes** an above-threshold finding.

### Audit trail

- Each cycle writes a structured log (e.g. `reports/adversarial_review.json`) containing, per finding: description, severity, status (`fixed` / `residual` / `escalated`), the round it first appeared, and -- for anything unfixed -- *why it was not fixed*.
- The log also records total rounds run and the stop reason (`converged` / `cap-reached` / `oscillation`).

### Configuration (`pyproject.toml`)

```toml
[tool.adversarial-review]
max_iterations = 5          # hard-cap backstop
fix_threshold = "high"      # fix Critical + High; Medium/Low are residual
report_path = "reports/adversarial_review.json"
```

## Pre-commit Configuration

Required `.pre-commit-config.yaml`:

```yaml
repos:
  # General file hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: <latest>
    hooks:
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
      - id: end-of-file-fixer
      - id: mixed-line-ending
      - id: trailing-whitespace

  # Secret detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: <latest>
    hooks:
      - id: detect-secrets

  # Python: lint + format
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: <latest>
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # Python: type checking (mypy)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: <latest>
    hooks:
      - id: mypy

  # Python: security (bandit)
  # Note: can be dropped if ruff S rules provide sufficient coverage
  - repo: https://github.com/PyCQA/bandit
    rev: <latest>
    hooks:
      - id: bandit
        args: [-r, -ll]

  # Python: SAST (fast ruleset only -- full scan in CI)
  - repo: https://github.com/returntocorp/semgrep
    rev: <latest>
    hooks:
      - id: semgrep
        args: ['--config', 'p/python', '--error']

  # Python: dependency vulnerabilities
  - repo: https://github.com/pypa/pip-audit
    rev: <latest>
    hooks:
      - id: pip-audit

  # Python: license compliance
  - repo: https://github.com/piber20/pre-commit-pip-licenses
    rev: <latest>
    hooks:
      - id: pip-licenses
        args: ['--fail-on', 'GPL-3.0']

  # Terraform / IaC (include only if project uses Terraform)
  # - repo: https://github.com/antonbabenko/pre-commit-terraform
  #   rev: <latest>
  #   hooks:
  #     - id: terraform_fmt
  #     - id: terraform_validate
  #     - id: terraform_tflint
  #     - id: terraform_trivy
  #     - id: terraform_docs
```

Run `pre-commit autoupdate` to pin to latest versions after initial setup.

## Git Commit Rules

- NEVER add any Claude signature, co-author lines, or "Generated with Claude" footers to git commit messages
- Always use conventional commit format: `type(scope): message`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
  - Scope: module or area affected (e.g., `auth`, `api`, `deps`, `tui`)
  - Examples: `feat(tags): add tag management commands`, `fix(deps): handle orphan references in cycle detection`
- Commit frequently in small, focused batches -- don't accumulate large changesets
- Before stopping work, check for pending changes. Run validation (tests/lint/build), fix any failures, then commit if all passes
- Consider enforcing with commitlint or commitizen (optional -- manual discipline is acceptable)

## Monorepo / Multi-Package Projects

When a project has multiple packages (e.g., `src/`, `web-ui/backend/`, `web-ui/frontend/`):

- Each package has its own `pyproject.toml` with independent deps, coverage targets, and tool config
- CI runs quality checks per-package in parallel
- Coverage targets may differ: 100% for core CLI, 90% for web API, 100% for frontend
- Shared test fixtures belong in the core package's `conftest.py`
- Cross-package integration tests live in a dedicated `tests/` directory or CI job

## Container Build Standards (when applicable)

- Multi-stage builds to minimize image size
- Non-root user in production stage
- `.dockerignore` / `.containerignore` to exclude dev files, `.git/`, `node_modules/`
- Pin base image digests for reproducibility
- Scan with trivy after build: `trivy image <image> --severity HIGH,CRITICAL --exit-code 1`
- Health check endpoint in the application

## Error Handling

- Define a project-level exception hierarchy rooted in a base exception (e.g., `class PTKError(Exception)`)
- User-facing errors: clear messages, no stack traces, appropriate exit codes
- Internal errors: raise typed exceptions, let them propagate, handle at CLI boundary
- Validate at system boundaries only (user input, external APIs, file parsing)
- Trust internal code and framework guarantees -- don't defensively check internal invariants

## Logging (when applicable)

- Use `logging` stdlib or `structlog` for structured logging
- Log levels: DEBUG for internal state, INFO for operations, WARNING for recoverable issues, ERROR for failures
- Never log secrets, tokens, or PII
- CLI tools: prefer stderr for diagnostic messages, stdout for data output
- JSON-structured logs for services/APIs; human-readable for CLI tools

## Documentation

- Docstring format: Google style (one-line summary, Args/Returns/Raises sections)
- Avoid multi-paragraph docstrings -- keep them to one or two lines maximum
- Don't document the obvious (what the code does) -- document the non-obvious (why, constraints, edge cases)
- API documentation: auto-generated from type hints + docstrings (mkdocs-material + mkdocstrings, or sphinx)
- README.md: required for every project root
- Architecture decisions: ADR format in `docs/adr/` (optional)

## CI/CD Pipeline Template

Minimum CI pipeline structure:

```
quality (ruff + mypy + pyright, ~30s)
    ├── test (pytest + coverage, ~2min)
    ├── security (semgrep + pip-audit + pip-licenses, ~1min)
    └── frontend (vitest/playwright, ~1min, if applicable)
          └── contracts (schema validation + backward compat, ~30s)

# Pre-PR (local, agent-driven, before CI):
adversarial-review (bounded review-fix loop -- see Automated Adversarial Code Review)

# On container changes:
container-scan (trivy)

# Scheduled nightly (non-blocking):
mutation (mutmut)
performance (pytest -m performance)
```

All pipeline stages must pass before merge. Nightly jobs report but don't block.

## Quality Gate Scripts

Every project should have OS-agnostic Python gate scripts (not shell scripts):

- `scripts/pre_ticket.py <ticket-id>` -- runs all checks before starting implementation
- `scripts/post_ticket.py <ticket-id>` -- runs all checks + ticket-specific validation after implementation
- `scripts/adversarial_review.py` -- runs the bounded adversarial review-fix loop (see Automated Adversarial Code Review)

These scripts use `subprocess.run()` with `sys.executable` for all tool invocations, ensuring cross-platform compatibility (Windows/Linux/macOS).
