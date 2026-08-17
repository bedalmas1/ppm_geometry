# Implementation Plan — Geometry of Predictive Process Monitoring Representations

Target venue: **ICPM 2027, ML4PM workshop**.
Source specification: [`process_geometry_experiment_prompt.md`](./process_geometry_experiment_prompt.md) (referred to below as "the spec"). This PLAN.md operationalizes the spec's Section 25 (Execution strategy) into concrete, checkable engineering phases. It is the static reference; day-to-day state lives in [`STATUS.md`](./STATUS.md) — **always read STATUS.md first when resuming work.**

## How these documents relate

- **PLAN.md** (this file): what the phases are, in what order, with deliverables and exit criteria. Edit this only when the plan itself changes (scope change, phase reordering, new decision constraints) — not to check off tasks.
- **STATUS.md**: current phase, per-task checkboxes, decision log, open questions, blockers. Edit this at the end of every work session.
- **CLAUDE.md**: onboarding pointer for a fresh Claude session — read STATUS.md, then the relevant phase section here.

## Governing principles (from spec §26)

Prefer, in order of priority: reproducibility over model count · strong baselines over many weak baselines · quantitative evidence over visual impressions · effect sizes over isolated p-values · controlled comparisons over architectural confounding · falsifiable hypotheses over confirmatory storytelling · simple metrics over unnecessary complexity · original latent dimensions over dimensionality-reduced coordinates.

Negative and null results are valid outputs, not failures. Do not force conclusions toward the "expected findings" listed in spec §17.

## Critical structural rule (spec §5)

Every experiment must be tagged as belonging to **Experiment Family A** (ecological/SOTA comparison — published models as-is) or **Experiment Family B** (controlled comparison — same encoder/embedding-dim/data/training-budget, objective varied). Never mix conclusions across families without saying so explicitly.

## Pilot-first rule (spec §27)

Do **not** scale to the full benchmark before Phase 6's pilot (1 dataset × 2 models, e.g. controlled Transformer-next vs Transformer-suffix) succeeds end-to-end: training → representation extraction → synthetic-validated geometry metrics → at least one descriptive result. Only after the pilot is validated does Phase 6 expand to the full model/dataset matrix.

---

## Phase 0 — Repository & tooling scaffolding

**Goal:** a reproducible Python project skeleton other phases build on.

Tasks:
- [ ] Choose environment/dependency manager (recommend `uv` for speed + lockfile discipline) and initialize `pyproject.toml` with `requires-python`, `[project]` metadata.
- [ ] Commit lockfile (`uv.lock`) once first real dependencies land in Phase 1/2 — do not pre-guess deep-learning framework versions before the literature audit picks the baselines (they may dictate e.g. PyTorch version compatibility).
- [ ] `.gitignore` for `data/raw`, `data/processed`, `results/`, model checkpoints, `__pycache__`, notebook checkpoints. Never gitignore lockfiles.
- [ ] `.python-version` / equivalent pin.
- [ ] `configs/` convention: one YAML per experiment (dataset + model + seed + hyperparams), consumed by a single `experiments/run_*.py` entrypoint. No hardcoded paths/hyperparams in source.
- [ ] `tests/` wired to a test runner (pytest); CI optional but at least a local `make test` / `scripts/test.sh`.
- [ ] Experiment tracking/logging decision (e.g. simple structured JSON/CSV logs under `results/` vs. an external tracker) — keep it lightweight; this is a research repo, not a product.
- [ ] README.md skeleton: project summary, how to reproduce, directory map, link to PLAN.md/STATUS.md.

**Deliverable:** installable repo skeleton, `pytest` runs (even if trivially, on a placeholder test) green.

**Exit criteria:** a fresh clone + documented setup command reaches a working environment.

---

## Phase 1 — Literature & repository audit

Maps to spec §25 Phase 1 and spec §4.

**Goal:** replace assumed model names in the spec with *verified* choices.

Tasks:
- [ ] Search recent (last ~3–5 years) peer-reviewed PPM literature for: next-event Transformer baselines (ProcessTransformer and any newer reproducible SOTA), full-suffix models (SuTraN/SuffixTransformerNetwork and alternatives), outcome-prediction models, self-supervised/representation-learning PPM baselines.
- [ ] For each candidate, build the table specified in spec §25: paper, year, venue, prediction task, architecture, datasets used, GitHub repo URL, license, last update date, reproducibility status (does it actually run?), accessible representation layer (can we hook out pre-head activations?), expected integration difficulty (S/M/L).
- [ ] Concretely inspect each shortlisted repo: clone, check it installs, identify the exact tensor/module that constitutes the prefix representation \(z_t\) before the final prediction head.
- [ ] Finalize model roster: aim for 3–6 configurations total, spanning Family A (real published models, ≥2: one next-event, one full-suffix, optionally one outcome model) and Family B (controlled Transformer encoder trained under ≥2 objectives — next-event, suffix, and outcome-if-defined-meaningfully — same architecture/dim/budget).
- [ ] Explicitly record why any spec-suggested model was rejected or replaced.
- [ ] Estimate compute budget (GPU-hours per model per dataset, total for full matrix) before committing to the final matrix size.

**Deliverable:** `paper/related_work_model_audit.md` (or `.csv` + write-up) with the comparison table; a finalized model roster written into STATUS.md's decision log.

**Exit criteria:** every model in the roster has a verified, runnable public implementation (or a justified from-scratch controlled implementation for Family B) and a known extraction point for \(z_t\).

---

## Phase 2 — Dataset pipeline

Maps to spec §6, §25 Phase 2.

Tasks:
- [ ] Select 4–6 public event logs (BPI Challenge candidates + others) covering variation in: case count, activity count, trace length, trace variability, branching, loops/rework, outcome diversity, structuredness. Do not cherry-pick for pretty geometry.
- [ ] Define outcome label(s) per dataset where used (documented, not implicit).
- [ ] Implement one shared preprocessing pipeline (`src/data/`) producing identical train/val/test splits consumed by every model — no per-model bespoke splitting.
- [ ] Guard against temporal/data leakage explicitly (e.g. time-based split where cases don't straddle boundaries incorrectly).
- [ ] Compute and record per-dataset descriptive stats required by spec §6: cases, events, activities, median/mean trace length, trace variants, variant entropy, relevant attributes, temporal span, outcome definition.
- [ ] Decide seed strategy (how many seeds, applied where compute allows — at minimum note where multi-seed was infeasible and why).

**Deliverable:** `src/data/` pipeline + `configs/datasets/*.yaml` + a dataset-statistics table/notebook feeding directly into the paper.

**Exit criteria:** every dataset produces identical, versioned (hash-recorded) splits usable by all models in the roster.

---

## Phase 3 — Baseline reproduction

Maps to spec §25 Phase 3, §10.

Tasks:
- [ ] Train/reproduce each Family A model on each dataset; compare against published numbers within reasonable tolerance.
- [ ] Train each Family B controlled-Transformer variant (next-event / suffix / outcome) with matched architecture, embedding dim, preprocessing, and training budget.
- [ ] Record standard predictive metrics per spec §10 (accuracy, macro-F1, NLL, ECE/calibration, Brier, DL-similarity for suffixes, remaining-time MAE, AUROC/AUPRC for outcome) using metrics standard to each model's original benchmark for comparability.
- [ ] Where reproduction fails or diverges from published numbers, document why (data version mismatch, split difference, undocumented hyperparameter, etc.) rather than silently adjusting until numbers match.
- [ ] Checkpoint every trained model with full provenance: dataset version/hash, config, seed, git commit, software versions, hardware.

**Deliverable:** checkpoints + `results/baseline_predictive_metrics.*` + reproduction notes.

**Exit criteria:** for each model/dataset pair, either (a) predictive metrics are within documented tolerance of literature, or (b) a documented, non-hand-wavy explanation of the gap exists.

---

## Phase 4 — Representation extraction

Maps to spec §7, §25 Phase 4.

Tasks:
- [ ] For each trained model, implement a hook that extracts \(z_t = f(e_{1:t})\) for every prefix of every test-set trace, from a documented, pre-prediction-head layer (CLS/state token, last-token, pooled encoder, decoder/context — explicitly named per model).
- [ ] Store per-trace \(Z_\sigma = (z_1,\dots,z_T)\) alongside: true events, predicted events/suffix, prediction probabilities, entropy, outcome, remaining time, prefix length, case metadata, correctness — in a standardized, cached format (e.g. one file/array per trace or a columnar store) so geometry analyses never require retraining.
- [ ] Version/hash the embedding cache against (model checkpoint, dataset split, extraction code) so stale caches are detectable.

**Deliverable:** `src/representations/` extraction code + cached embedding datasets under `data/processed/embeddings/` (or `results/embeddings/`, per Phase 0's convention).

**Exit criteria:** embeddings for the pilot model pair (Phase 6) are extracted, cached, and reloadable without touching the trained model again.

---

## Phase 5 — Geometry metric implementation & synthetic validation

Maps to spec §8, §9, §22, §25 Phase 5. **Do this before analyzing any real model.**

Tasks:
- [x] Implement modular `src/geometry/` package: straightness (§8.1), path length (§8.2), velocity/displacement (§8.3), curvature (§8.4, with careful handling of near-zero-norm degeneracies), smoothness/acceleration (§8.5), progress-to-terminal-region (§8.6, Euclidean prototype + at least one neighborhood/distribution-based alternative), branch separation over normalized progress (§8.7), future-equivalence neighborhood quality (§8.8: edit-distance/Damerau-Levenshtein/activity-set/n-gram/outcome/remaining-time distances vs. latent distance, rank correlation, precision@k, trustworthiness/continuity).
- [x] Implement representation-quality diagnostics required by §9: embedding variance, covariance spectrum, effective rank, pairwise-distance distribution, intrinsic dimensionality, terminal-state separability, neighborhood preservation. These must always accompany trajectory metrics — never report straightness without them.
- [x] Build the synthetic trajectory suite from spec §22: perfectly straight, circular-arc curved, backtracking, collapsed (\(z_t=c\)), branching, reconverging. Assert analytically-known expected values (e.g. straightness = 1 for the straight case, collapse triggers degenerate-geometry flags).
- [x] Write `tests/` for every metric against the synthetic suite — this is the primary regression protection for the whole project.

**Deliverable:** `src/geometry/` + `tests/test_geometry_*.py`, all green against synthetic ground truth. DONE — see [`paper/phase5_geometry_metrics.md`](./paper/phase5_geometry_metrics.md).

**Exit criteria:** every metric has at least one synthetic case with a known correct value, and passes. MET (44 tests, all green).

---

## Phase 6 — Descriptive geometry study (pilot → full scale)

Maps to spec §25 Phase 6, RQ1, H1–H3.

**Pilot sub-phase (mandatory gate before scaling):**
- [ ] Run the full pipeline (train → extract → geometry metrics) on **one dataset × two models** (recommend the two controlled-Transformer objective variants from Family B, since that isolates objective from architecture cleanly).
- [ ] Validate that representation extraction and geometry metrics behave sanely on real (non-synthetic) data — sanity-check against the diagnostics from Phase 5/§9 before trusting any trajectory metric.
- [ ] Confirm compute/time budget per model/dataset is in line with Phase 1's estimate; revise the full-matrix plan if not.

**Full-scale sub-phase (only after pilot passes):**
- [ ] Run the complete model × dataset matrix (from Phases 1–4) through the geometry package.
- [ ] Characterize RQ1 properties across all combinations: straightness, smoothness, curvature, displacement, velocity, efficiency, clustering, branch structure, terminal-state organization, intrinsic dimensionality, local neighborhood structure.
- [ ] Explicitly test H1 (suffix > next-event global organization), H2 (next-event locally discriminative but globally fragmented), H3 (outcome supervision → strong terminal clustering, weaker intermediate structure) — report support/rejection for each, not just descriptive numbers.
- [ ] Keep Family A vs Family B comparisons visually and statistically separate throughout (§5).

**Deliverable:** `results/geometry_descriptive/*`, first batch of publication figures (types 1–4, 10 from spec §16).

**Exit criteria:** every model/dataset combination has a full geometry + diagnostics report; H1–H3 have an explicit verdict.

---

## Phase 7 — Reliability study

Maps to spec §11, RQ2/RQ3, H4. **Spec marks this as one of the project's highest-priority experiments.**

Tasks:
- [ ] Assemble per-prediction feature sets: baseline reliability (max prob, entropy, margin, prefix length) and geometric features (straightness-so-far, recent/max curvature, recent displacement, cumulative path length, smoothness, distance-to-training-manifold / NN distance, local density, distance to predicted outcome/future region).
- [ ] Fit the three nested models on validation data: **A** — P(error | confidence); **B** — P(error | confidence, prefix length); **C** — P(error | confidence, prefix length, geometry). Keep these interpretable (e.g. logistic regression / calibrated simple classifiers), not opaque.
- [ ] Evaluate on held-out test data: AUROC, AUPRC, Brier score, calibration, likelihood improvement (A→B→C), with paired statistical tests and bootstrap confidence intervals.
- [ ] Specifically analyze high-confidence errors (the case conventional confidence measures miss).
- [ ] Verdict: does geometry provide incremental information about failure beyond confidence + prefix length? Report the effect size, not just significance.

**Deliverable:** `results/reliability/*`, reliability comparison figures (spec §16 item 7).

**Exit criteria:** explicit, quantified answer to spec §24 Q4, with CIs, across all datasets/models tested.

---

## Phase 8 — Future-organization study

Maps to spec §8.8, §25 Phase 8, RQ5, H6/H7.

Tasks:
- [ ] Define future-similarity ground truth independent of the latent space (edit distance / Damerau-Levenshtein / activity-set / n-gram / outcome / remaining-time similarity).
- [ ] For prefix pairs, compare latent distance \(d_Z\) to future distance \(d_F\): rank correlation, nearest-neighbor retrieval quality, precision@k, trustworthiness/continuity — controlling for history similarity.
- [ ] Run the two decisive tests from spec §8.8: (a) among prefixes with *dissimilar histories*, does latent proximity retrieve similar-future cases? (b) among prefixes with *similar histories*, does latent distance increase as futures diverge?
- [ ] Measure branch separation over normalized prefix progress (§8.7); only define a "geometric predictive horizon"/bifurcation time if the statistics robustly support it — otherwise explicitly decline to define one.
- [ ] Test H6 (do models differ in how early they separate divergent-future cases?) and H7 (does organization reflect future similarity beyond mere prefix/activity similarity in the strongest models?).

**Deliverable:** `results/future_organization/*`, branch-separation and future-similarity-vs-latent-distance figures (spec §16 items 6, 9).

**Exit criteria:** explicit verdicts on H6/H7 and spec §24 Q6.

---

## Phase 9 — Explainability / event-importance study

Maps to spec §12–13, §25 Phase 9, RQ4, H5.

Tasks:
- [ ] For every event transition, compute displacement, direction, curvature, and movement relative to terminal/future regions; rank events by geometric impact.
- [ ] Quantitatively test whether high-impact events associate with: outcome-differentiating activity transitions, rework, deviations, rare events, known process milestones, future suffix changes — statistical association, not cherry-picked qualitative examples.
- [ ] Where feasible, compare geometric event importance against an established baseline (attention scores, gradient attribution, or SHAP) — comparison, not replacement.
- [ ] Distinguish **trajectory turning points** (\(\theta_t \gg 0\)) from **future turning points** (post-event shift in observed future-outcome distribution) as in spec §13; do not assume they coincide — test the relationship explicitly.
- [ ] Use non-causal language throughout ("representation shift", "model-internal turning point") per spec §12's explicit constraint.

**Deliverable:** `results/explainability/*`, annotated event-level curvature/displacement figures (spec §16 item 8).

**Exit criteria:** quantified association tests (not just plots) backing H5 and spec §24 Q5.

---

## Phase 10 — Controls, ablations & statistical synthesis

Maps to spec §14–15, §25 Phase 10.

Tasks:
- [ ] Run required controls: random/untrained encoder; alternative representation layer (≥1 architecture); normalization robustness (raw / L2 / whitened); distance metric (Euclidean vs cosine, at minimum); prefix-length confound check; trace-length confound check; activity-frequency confound check; multi-seed stability (as compute allows).
- [ ] Aggregate all findings across datasets/models with bootstrap CIs, effect sizes, multiple-comparison correction, and mixed-effects/regression modeling where dataset introduces hierarchical structure.
- [ ] Explicitly test cross-dataset replication of every major finding (spec §24 Q7) — report where findings do *not* replicate.
- [ ] Assess which geometric metrics were actually informative vs. redundant/misleading (spec §24 Q8).

**Deliverable:** `results/statistical_synthesis/*`, ablation tables.

**Exit criteria:** every H1–H7 and RQ1–RQ5 has a dataset-level replication verdict, not just a pooled one.

---

## Phase 11 — Paper

Maps to spec §23, §25 Phase 11.

Tasks:
- [ ] Maintain the continuously-updated research report (spec §23 sections 1–13) as the paper draft evolves — treat it as living documentation, not a one-shot writeup at the end.
- [ ] Generate every table/figure programmatically from `results/` — no hand-copied numbers into `paper/`.
- [ ] Explicitly answer spec §24 Q1–Q10 as a dedicated results/discussion subsection, including willingness to answer Q9/Q10 with "not yet" if unsupported.
- [ ] Write threats-to-validity and rejected-hypotheses sections honestly (spec explicitly requires reporting rejected hypotheses).
- [ ] Target ICPM 2027 ML4PM workshop formatting/length constraints (confirm CFP details closer to submission — not yet verified as of plan authoring).

**Deliverable:** submission-ready manuscript in `paper/`.

**Exit criteria:** manuscript complete, all figures/tables regenerate from a clean `results/` run, Q1–Q10 answered.

---

## Cross-cutting reminders (apply in every phase)

- Never use UMAP/t-SNE coordinates for quantitative claims — original latent space only (visualization use is fine).
- Never claim straightness is inherently good, Euclidean geometry is necessarily correct, displacement is causal, or UMAP clusters prove organization (spec §18 — full list).
- Tag every experiment Family A or Family B (spec §5).
- Every trained artifact needs: dataset hash, preprocessing version, split, config, seed, software versions, checkpoint, git commit.
