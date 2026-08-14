# Config convention

One YAML file per experiment run, consumed by a single entrypoint under `experiments/`. No hardcoded paths, hyperparameters, or seeds in source code — everything an experiment needs to be reproduced lives in its config.

Planned subdirectories (created as each phase needs them):

- `configs/datasets/` — one file per event log: source location, preprocessing options, split definition.
- `configs/models/` — one file per model configuration (architecture + hyperparameters), tagged as Family A (published model, as-is) or Family B (controlled comparison).
- `configs/experiments/` — combines a dataset config + model config + seed(s) + any experiment-specific settings (e.g. which geometry metrics to run).

Every config used to produce a checkpoint or result must be saved alongside that artifact's provenance record (dataset hash, split, seed, software versions, git commit) — see PLAN.md's cross-cutting reminders.
