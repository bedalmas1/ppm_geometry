# ppm-geometry

What geometric structure do predictive process monitoring (PPM) models learn in their latent representations, and does that geometry tell us something about process behavior, prediction reliability, and explainability beyond conventional predictive metrics?

Research project targeting **ICPM 2027 — ML4PM workshop**. Full research specification: [`process_geometry_experiment_prompt.md`](./process_geometry_experiment_prompt.md).

## Start here

- [`STATUS.md`](./STATUS.md) — current progress, decision log, open questions. Read this first.
- [`PLAN.md`](./PLAN.md) — the full phased implementation plan (Phase 0 → Phase 11).
- [`CLAUDE.md`](./CLAUDE.md) — onboarding notes for AI-assisted sessions working in this repo.

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync              # creates .venv and installs locked dependencies
uv run pytest -q     # run the test suite
```

If `uv` isn't on your PATH, `python -m uv <command>` works identically.

## Repository layout

```text
configs/            experiment configuration YAMLs (dataset + model + seed + hyperparams)
data/
  raw/              untouched source event logs (gitignored; not committed)
  processed/        derived splits/caches (gitignored; regenerate via src/data)
src/
  data/             dataset loading, preprocessing, splitting
  models/           model implementations / wrappers around reproduced baselines
  representations/  prefix-embedding extraction (z_t = f(e_1:t)) and caching
  geometry/         trajectory/geometry metrics (straightness, curvature, ...) + synthetic validation
  evaluation/       predictive metrics (accuracy, F1, calibration, suffix similarity, ...)
  reliability/      error-prediction models (confidence vs. confidence+geometry)
  explainability/   event-importance / turning-point analysis
  visualization/    figure generation (PCA/UMAP for display only, never for quantitative claims)
experiments/        run entrypoints, one per experiment, config-driven
scripts/            one-off utility scripts
notebooks/          exploratory notebooks (not a substitute for src/ + tests/)
results/            experiment outputs (gitignored; regenerable from configs + code)
figures/            generated publication figures (gitignored; regenerable)
tests/              pytest suite, including the geometry-metric synthetic validation suite
paper/              manuscript source and the model/dataset audit tables
```

Every trained artifact and cached embedding set should carry provenance: dataset version/hash, preprocessing version, split, model config, random seed, software versions, checkpoint path, and git commit — see PLAN.md's cross-cutting reminders.

## Status

No models or datasets are implemented yet. See STATUS.md for the current phase and next steps.
