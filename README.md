# Colleague Candidates

[![CI](https://github.com/AI-Colleagues/colleague-candidates/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/AI-Colleagues/colleague-candidates/actions/workflows/ci.yml?query=branch%3Amain)

This repository hosts candidate AI colleagues for Orcheo. Each colleague is an
Orcheo workflow stored under `colleagues/`, and is surfaced in the Canvas
Candidates tab, which reads the `main` branch of this repository.

## Structure

Each colleague lives in its own directory under `colleagues/` containing:

- `workflow.py` — the workflow definition, prefixed with a `# /// orcheo`
  frontmatter block (`name`, `handle`, `description`, `entrypoint`, and
  optional `emoji` / `subtitle` / `notes` / `[metadata]`).
- `config.json` — an optional companion runnable config.

The Canvas Candidates tab reads every colleague from the `main` branch of
this repository.
