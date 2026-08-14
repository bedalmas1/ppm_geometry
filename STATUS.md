# Project Status

> Living tracker. Update this file at the end of every session — new sessions should read this file first, then jump to the relevant phase in [`PLAN.md`](./PLAN.md). Keep entries dated; never delete history from the decision log, only append.

**Current phase:** Phase 1 — Literature & repository audit (not started)
**Last updated:** 2026-08-14

---

## Phase checklist (mirrors PLAN.md — tick here as work completes)

- [x] Phase 0 — Repository & tooling scaffolding
- [ ] Phase 1 — Literature & repository audit
- [ ] Phase 2 — Dataset pipeline
- [ ] Phase 3 — Baseline reproduction
- [ ] Phase 4 — Representation extraction
- [ ] Phase 5 — Geometry metric implementation & synthetic validation
- [ ] Phase 6 — Descriptive geometry study (pilot → full scale)
- [ ] Phase 7 — Reliability study
- [ ] Phase 8 — Future-organization study
- [ ] Phase 9 — Explainability / event-importance study
- [ ] Phase 10 — Controls, ablations & statistical synthesis
- [ ] Phase 11 — Paper

---

## Current phase detail

### Phase 0 — Repository & tooling scaffolding — DONE
- Dependency manager: `uv` (installed via `pip install --user uv`; not on this machine's PATH inside the bash tool, so `python -m uv <cmd>` was used — works identically; a normal shell with `uv`'s install location on PATH will have `uv` directly).
- `pyproject.toml`: project metadata, `requires-python = ">=3.11,<3.14"` (permissive placeholder — Phase 1 may need to narrow this if a chosen baseline repo requires an older torch/Python combination), `[tool.uv] package = false` (this is a research repo, not a distributable package — imports rely on `pythonpath = ["src"]` in pytest config, no editable install needed), `dev` dependency group with `pytest`.
- `uv.lock` generated and committed.
- `.venv/` created locally (gitignored).
- `.gitignore` added: excludes `.venv/`, caches, `data/raw/*` and `data/processed/*` (keeps `.gitkeep`), `results/*` and `figures/*` (keeps `.gitkeep`), checkpoint file extensions, notebook checkpoints, editor/OS cruft. Explicitly keeps `uv.lock`, `PLAN.md`, `STATUS.md`.
- `tests/test_scaffold.py` placeholder + `pytest` config (`testpaths`, `pythonpath`) — `uv run pytest -q` passes (1 test). Removed `tests/.gitkeep` since a real file now lives there.
- `configs/README.md` documents the config convention (one YAML per experiment; `datasets/`, `models/`, `experiments/` subdirectories to be created as needed) — replaced `configs/.gitkeep`.
- `README.md` written: project summary, setup instructions, full repo layout, pointers to STATUS/PLAN/CLAUDE.

Not yet decided in Phase 0 (deferred, not blocking): experiment tracking/logging format (lightweight structured logs vs. external tracker) — will decide once Phase 3 (baseline reproduction) actually needs to log training runs.

### Phase 1 — Literature & repository audit
Not started. See PLAN.md Phase 1 for the full task list.

---

## Decision log

Append one entry per decision, dated, with a one-line rationale. Never edit past entries — supersede them with a new dated entry if a decision changes.

- **2026-08-14** — Created PLAN.md/STATUS.md/CLAUDE.md tracking system and empty reproducibility directory skeleton per spec §21. No implementation work started yet.
- **2026-08-14** — Chose `uv` as the Python dependency manager (user confirmed over pip+pip-tools and poetry). Installed via `pip install --user uv` since it wasn't preinstalled.
- **2026-08-14** — Completed Phase 0: pyproject.toml (`requires-python>=3.11,<3.14`, non-package project via `[tool.uv] package = false`), uv.lock, .gitignore, pytest wiring with a passing placeholder test, configs/README.md, and README.md. Deferred the experiment-tracking-format decision to Phase 3 (not needed until real training runs exist).

---

## Open questions / not yet decided

- Final model roster (Family A + Family B) — pending Phase 1 literature audit; spec's suggested names (ProcessTransformer, SuTraN) are unverified starting points only.
- Final dataset roster (4–6 logs) — pending Phase 2, informed by Phase 1's compute estimate.
- Experiment tracking/logging format (lightweight structured logs under `results/` vs. an external tracker) — deferred until Phase 3 needs to log actual training runs.
- Number of seeds per model/dataset — depends on compute budget, to be estimated in Phase 1.
- `requires-python` upper/lower bound in `pyproject.toml` is a placeholder (`>=3.11,<3.14`) — may need narrowing once Phase 1 picks concrete baseline repos with their own Python/torch constraints.
- ICPM 2027 ML4PM workshop CFP details (page limit, format, deadline) — not yet looked up; needed before Phase 11.

## Blockers

None currently.

## Next steps (pick these up first in the next session)

1. Begin Phase 1 literature/repository audit (see PLAN.md Phase 1 tasks) — this is the mandatory first substantive research step per spec §27; do not start implementing models before it.
2. As candidate model repos are identified, revisit the `requires-python` placeholder in `pyproject.toml` if any repo forces a narrower Python/torch version.
