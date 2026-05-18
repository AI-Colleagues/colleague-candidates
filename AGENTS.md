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

Workflow metadata should stay consistent with the frontmatter keys used in existing files: `name`, `handle`, `description`, `entrypoint`, and optional `emoji`, `subtitle`, and `config`.

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
