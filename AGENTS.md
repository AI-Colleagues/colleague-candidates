# Repository Guidelines

## Project Structure & Module Organization
This repository stores Orcheo colleague workflows, one per directory under `colleagues/`. Each colleague typically contains:

- `workflow.py` for the workflow definition and `# /// orcheo` frontmatter.
- `config.json` for optional runnable configuration.

Supporting planning and design notes live under `docs/`, with reusable templates in `docs/templates/`. Keep new colleagues in `colleagues/<handle>/` using snake_case directory names that match the workflow handle, while the handle itself should be in kebab-case.

### Self-contained workflows
Each `workflow.py` is intentionally self-contained. Code duplication across workflows (helpers, node subclasses, utility functions) is acceptable and expected — do not extract shared code into a common module. This keeps each workflow independently deployable and readable without cross-file dependencies.

## Build, Test, and Development Commands
This project uses `uv`-managed Python 3.12 dependencies.

- `make lint` runs `ruff check .` and `ruff format . --check`.
- `make format` applies `ruff format .` and fixes import/order issues.
- `uv sync` installs the locked dependencies from `uv.lock`.
- `uv run ruff check .` is a direct alternative to the Make target.

There is no dedicated automated test suite in the repository yet, so linting is the primary validation step.

## Coding Style & Naming Conventions
Follow the existing Python conventions enforced by `pyproject.toml`:

- Use 88-character lines.
- Prefer Google-style docstrings for public functions and classes.
- Keep imports sorted and grouped by `ruff`/`isort`.
- Use type annotations on public functions; `ruff` is the sole linting tool.
- Use snake_case for Python modules and functions.

Workflow metadata should stay consistent with the frontmatter keys used in existing files: `name`, `handle`, `description`, `entrypoint`, and optional `avatar`, `subtitle`, and `config`.

## Colleague Update Procedure
When updating an existing colleague workflow, treat the change as a released
candidate version so onboarded colleagues can show update availability and
release notes in Orcheo Studio.

- Keep the existing `handle` stable unless the requested change is explicitly a
  new colleague. Changing the handle breaks the relationship with already
  onboarded colleagues.
- Add or bump the frontmatter `version` field using strict SemVer
  `MAJOR.MINOR.PATCH` with no leading `v`, prerelease suffix, or build metadata.
- Add a `[[updates]]` frontmatter entry for the new version with a concise
  `summary` and, when operator action may be needed, a `migration` note.
- Use patch versions for compatible bug fixes or prompt/config refinements,
  minor versions for backward-compatible new capability, and major versions for
  behavior, configuration, credential, or output-shape changes that may require
  review before upgrading.
- Preserve existing `config.json` keys and secret placeholders whenever possible.
  If a config key, credential placeholder, or output contract changes, call it
  out in the `migration` note.
- Keep `updates` focused on user-visible changes since each entry is shown in
  the Studio update flow. Do not use it for internal refactors that do not affect
  operators or downstream workflows.
- Run `make lint` before handoff. If the workflow has a local runnable config or
  targeted test coverage, run the relevant `uv run` validation as well.

Example frontmatter:

```python
# /// orcheo
# name = "Insight Analyst"
# handle = "insight-analyst"
# description = "Analyzes research inputs and produces concise insight summaries."
# version = "1.3.0"
#
# [[updates]]
# version = "1.3.0"
# summary = "Adds source-quality checks before insight synthesis."
# migration = "Review custom prompt overrides if they assume every source is accepted."
# ///
```

## Testing Guidelines
No `tests/` directory exists today. If you add tests, place them under `tests/` and use `test_*.py` naming. Prefer focused checks around workflow assembly, config parsing, and any helper logic.

## Commit & Pull Request Guidelines
Recent commits use short imperative summaries such as `Update orcheo version` and `Restructure templates into colleagues/ and add PEP 723 frontmatter`. Keep commits narrow and descriptive.

Pull requests should include:

- A short description of the workflow or docs change.
- Any related issue or discussion link.
- Screenshots or sample output only when the change affects generated docs or UI-facing behavior.

## Security & Configuration Tips
Do not commit real secrets. Existing workflows reference secrets with placeholders like `[[openai_api_key]]`; keep that pattern and store actual values in the Orcheo vault or runtime config.
