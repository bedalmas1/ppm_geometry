# Project Status

> Living tracker. Update this file at the end of every session — new sessions should read this file first, then jump to the relevant phase in [`PLAN.md`](./PLAN.md). Keep entries dated; never delete history from the decision log, only append.

**Current phase:** Phase 0 — Repository & tooling scaffolding
**Last updated:** 2026-08-14

---

## Phase checklist (mirrors PLAN.md — tick here as work completes)

- [ ] Phase 0 — Repository & tooling scaffolding
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

### Phase 0 — Repository & tooling scaffolding
Not started. Directory skeleton exists (`configs/`, `data/{raw,processed}/`, `src/{data,models,representations,geometry,evaluation,reliability,explainability,visualization}/`, `experiments/`, `scripts/`, `notebooks/`, `results/`, `figures/`, `tests/`, `paper/`, each with a `.gitkeep`). No `pyproject.toml`, no `.gitignore`, no README yet — these are next.

---

## Decision log

Append one entry per decision, dated, with a one-line rationale. Never edit past entries — supersede them with a new dated entry if a decision changes.

- **2026-08-14** — Created PLAN.md/STATUS.md/CLAUDE.md tracking system and empty reproducibility directory skeleton per spec §21. No implementation work started yet.

---

## Open questions / not yet decided

- Dependency manager for Python (`uv` recommended in PLAN.md Phase 0, not yet confirmed with user).
- Final model roster (Family A + Family B) — pending Phase 1 literature audit; spec's suggested names (ProcessTransformer, SuTraN) are unverified starting points only.
- Final dataset roster (4–6 logs) — pending Phase 2, informed by Phase 1's compute estimate.
- Experiment tracking tool (lightweight structured logs vs. e.g. an external tracker) — not yet decided.
- Number of seeds per model/dataset — depends on compute budget, to be estimated in Phase 1.
- ICPM 2027 ML4PM workshop CFP details (page limit, format, deadline) — not yet looked up; needed before Phase 11.

## Blockers

None currently.

## Next steps (pick these up first in the next session)

1. Confirm/settle Phase 0 open questions above (dependency manager, tracking approach) with the user if not already decided.
2. Scaffold `pyproject.toml`, `.gitignore`, README.md, and a minimal passing test to close out Phase 0.
3. Begin Phase 1 literature/repository audit (see PLAN.md Phase 1 tasks) — this is the mandatory first substantive research step per spec §27; do not start implementing models before it.
