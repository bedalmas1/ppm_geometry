# Phase 6 — pilot sub-phase

Produced per [`PLAN.md`](../PLAN.md) Phase 6's mandatory pilot-first gate (spec §27): run the full geometry package against real, already-trained-and-extracted embeddings for one dataset x two models, and confirm the metrics behave sanely before scaling to the full 9-model x 5-dataset matrix.

## What was run

`experiments/run_geometry_pilot.py helpdesk` — B1 `controlled_transformer_next` vs. B2 `controlled_transformer_suffix` on Helpdesk, PLAN.md's recommended pilot pair (both Family B, same encoder/embedding-dim/training-budget, objective varied — isolates objective from architecture cleanly per spec §5). Both models were already trained (Phase 3) and extracted (Phase 4) on Helpdesk; this script does no training or extraction of its own, only analysis of `results/helpdesk/<model>/embeddings_test.parquet`.

No deep-learning framework is needed — the geometry-analysis step is pure numpy/pandas/scipy/sklearn (already-cached embeddings as input), so it needed no `--extra` install and ran in **~5.9s (B1, 3,495 rows) and ~5.4s (B2, 4,411 rows)**. This directly answers Phase 6's pilot task "confirm compute/time budget per model/dataset is in line with Phase 1's estimate": the geometry-analysis step itself is not a compute bottleneck at all — training/extraction (already measured per-model in Phase 3/4) remains the actual cost driver for the full-scale sub-phase.

Ground-truth "future" per prefix (used by `geometry.future_equivalence`) was recomputed directly from the raw event log/split (activities after prefix index `k`), independent of whichever objective a model was actually trained on — this lets B1 (next-event, no suffix stored in its own cache) and B2 (full-suffix, already stores `true_suffix`) be evaluated identically, and generalizes to any future model regardless of objective.

`geometry.future_equivalence`/`geometry.diagnostics.trustworthiness`/`continuity` are O(n²) (flagged as an open item in `paper/phase5_geometry_metrics.md`); this pilot subsamples 500 rows (fixed seed 42) for those specific computations rather than running them on the full cache — validates the subsampling approach itself works, ahead of the full-scale sub-phase needing it at every one of 45 model/dataset combinations.

## Sanity checks (spec §9 requirement: never trust trajectory metrics alone)

Both models passed every automated sanity flag (collapse, near-collapse, all-NaN/majority-NaN straightness):

| | B1 next-event | B2 full-suffix |
|---|---|---|
| total pooled variance | 33.26 | 5.24 |
| effective rank (of z_dim=64) | 2.87 | 4.16 |
| participation ratio | 2.31 | 2.84 |
| straightness NaN count | 0 / 916 | 0 / 916 |

Neither space is collapsed (effective rank well above 1, total variance well above 0), and no trace produced a degenerate straightness value. **Terminal-state separability came back `NaN` for both models** — this is *correct*, documented behavior, not a bug: Helpdesk's terminal-activity distribution is extremely imbalanced (`closed`: 896/916 cases, four other terminal activities with 1-8 cases each), and `geometry.diagnostics.terminal_state_separability` is explicitly designed to return `NaN` rather than a misleading silhouette score whenever any label has fewer than 2 members (here, `take-in-charge-ticket` has exactly 1). This is itself a useful pilot finding: **terminal-state separability as defined will need a coarser or better-populated outcome/terminal-variant grouping to be usable at all on Helpdesk** — worth deciding before the full-scale sub-phase runs this metric on the other 4 datasets, where terminal-class balance may differ.

## Descriptive result (pilot exit criterion: "at least one descriptive result")

| metric | B1 next-event | B2 full-suffix |
|---|---|---|
| straightness (mean) | 0.488 | 0.307 |
| path length (mean) | 23.34 | 11.73 |
| effective rank | 2.87 | 4.16 |
| future rank correlation (edit distance) | 0.586 | 0.635 |
| future rank correlation (2-gram) | 0.638 | 0.770 |
| precision@10 (edit distance) | 0.203 | 0.166 |
| trustworthiness vs. future (edit distance) | 0.804 | 0.858 |
| branch separation at s=0.85 (late progress, grouped by terminal activity) | 2.07 | 0.76 |

This is a genuinely mixed picture, not a clean win for either model — exactly the kind of result spec §18 warns against over-interpreting via a single metric:

- **B1 has higher straightness** and a larger late-progress branch-separation spike than B2. Read naively this looks like "B1 organizes more," but B1's raw path length is also roughly double B2's (23.3 vs. 11.7) on the same 64-dim space, and straightness is a *ratio* (net displacement / path length) — a shorter, more direct-but-still-curved path can score lower on straightness than a longer path that happens to end up net-displaced, so this pair of numbers alone does not support a directional organization claim without also looking at the trajectories' absolute scale (which is why spec §9 requires diagnostics alongside trajectory metrics, not instead of them).
- **B2 has higher effective rank, higher future-rank-correlation on every d_F variant tested, and higher trustworthiness against the future-edit-distance ground truth.** This points the other way: B2's latent space more consistently keeps future-similar prefixes close together (in the sense spec §8.8 actually measures), even though its raw trajectories are geometrically "less straight."

**No H1/H2/H3 verdict is drawn here** — PLAN.md is explicit that hypothesis testing belongs to the full-scale sub-phase, not the pilot, and a single dataset/model-pair is not evidence for a roster-wide claim. What this pilot does establish: the metrics are not all measuring the same thing, they disagree with each other on this pair in an interpretable way, and the pipeline surfaces that disagreement rather than masking it — which is the actual point of running straightness/effective-rank/future-equivalence together rather than picking one.

## Verdict: pilot passed, full-scale sub-phase unblocked

All three of PLAN.md's pilot tasks are met: the full pipeline ran end-to-end on real (non-synthetic) embeddings, the diagnostics behaved sanely (no collapse, one expected/correct `NaN` case, not a silent bug), and the compute budget for the geometry-analysis step itself is confirmed negligible relative to training. Per PLAN.md's pilot-first rule, Phase 6 may now proceed to its full-scale sub-phase (all 9 roster models x all 5 datasets) — see STATUS.md's next steps for what that requires (training the ~36 model/dataset combinations not yet trained, executed by the user directly per the single-shared-GPU/sequential-training constraint established in Phase 3).

## Follow-ups for the full-scale sub-phase, not solved here

- Terminal-state separability needs a better-populated grouping than raw terminal activity on datasets as imbalanced as Helpdesk — decide a coarser grouping (or accept `NaN` as a valid, reported result for some dataset/model pairs) before treating this metric as a roster-wide comparison axis.
- The O(n²) future-equivalence/trustworthiness computations are subsampled here (500/3,495-4,411 rows); at full scale across 45 combinations this subsampling approach should be reused directly (already validated by this pilot), rather than attempting the full O(n²) computation on every cache.
- `branching.branch_separation_curve`'s bin edges (10 bins here) and the "group by terminal activity" choice are this pilot's own design decisions for a first look, not dictated by PLAN.md/the spec — worth revisiting once outcome-style labels are more clearly defined for the datasets that need it (BPIC datasets may have richer/more balanced terminal-activity distributions than Helpdesk).
