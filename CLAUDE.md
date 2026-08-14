# ppm_geometry — session onboarding

This repo implements the research project specified in [`process_geometry_experiment_prompt.md`](./process_geometry_experiment_prompt.md): a study of the latent geometry learned by predictive process monitoring (PPM) models, targeting the ICPM 2027 ML4PM workshop.

**Every session must start by reading, in order:**

1. [`STATUS.md`](./STATUS.md) — current phase, completed work, decision log, open questions, blockers, and explicit "next steps." This is the source of truth for where the project actually is.
2. [`PLAN.md`](./PLAN.md) — the full phased implementation plan (Phase 0 through Phase 11) that STATUS.md's phases refer to. Read the section for the current phase (and the pilot-first rule under "Pilot-first rule" before touching Phase 6).
3. `process_geometry_experiment_prompt.md` — the original research specification. PLAN.md operationalizes it; when in doubt about scientific intent (not engineering sequencing), this file is authoritative.

**At the end of every session that makes progress**, update `STATUS.md`: tick completed phase-checklist items, append a dated decision-log entry for any non-obvious choice made, update "Open questions"/"Blockers," and rewrite "Next steps" to reflect what should happen next. Do not edit `PLAN.md` casually — it only changes when the plan itself changes (new phase, reordering, scope change), not to record day-to-day progress.

## Non-negotiable constraints from the research spec

- Do not start implementing/training models before the Phase 1 literature & repository audit is done (spec §27). The spec's suggested model names (ProcessTransformer, SuTraN, ...) are unverified starting points, not commitments.
- Do not scale to the full model×dataset matrix before the Phase 6 pilot (1 dataset × 2 models) succeeds end-to-end.
- Tag every experiment as Family A (ecological/SOTA, published models as-is) or Family B (controlled, same architecture/dim/budget, objective varied) — never blend conclusions across families (spec §5).
- All primary geometric metrics are computed in the original latent space; UMAP/t-SNE are for visualization only, never for quantitative claims (spec §7, §14).
- Negative/null results and rejected hypotheses must be reported, not hidden (spec §9's H1–H7 in the prompt file, §17–18).
- Every trained artifact needs recorded provenance: dataset hash, split, config, seed, software versions, checkpoint, git commit (spec §21).

## Repo layout

Standard structure from spec §21: `configs/` (experiment YAMLs), `data/{raw,processed}/`, `src/{data,models,representations,geometry,evaluation,reliability,explainability,visualization}/`, `experiments/` (run entrypoints), `scripts/`, `notebooks/`, `results/`, `figures/`, `tests/`, `paper/`. Directories currently exist as empty scaffolds (`.gitkeep` placeholders) — see STATUS.md Phase 0 for what's actually implemented.
