#!/usr/bin/env python
"""Phase 6 full-scale sub-phase: run the geometry package (src/geometry/)
against every already-trained-and-extracted model for one dataset, not just
the mandatory pilot pair.

Reuses run_geometry_pilot.py's per-model analysis (run_model,
load_raw_test_activities) unchanged - that function was already generic over
a single (dataset, model) pair; only the pilot script's *comparison* step was
hardcoded to exactly two models. Results are written to a separate
`full_scale/` output directory so they never overwrite the pilot's own
`pilot/` artifacts (PLAN.md's pilot sub-phase report must stay as-is).

Usage:
    uv run python experiments/run_geometry_full_scale.py [dataset] [model ...]

With no models given, auto-discovers every model under results/<dataset>/
that has a completed embeddings_test.parquet (skips in-progress training
directories that only hold a checkpoint/resume_state so far).

This is still purely descriptive (spec's Phase 6 scope, not H1-H3 hypothesis
verdicts or cross-dataset statistical synthesis - that's Phase 7/10) and pure
numpy/pandas/scipy/sklearn - no deep-learning framework, no GPU - so it's
safe to run alongside an in-progress training job on the same machine.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_geometry_pilot import KNN_K, load_raw_test_activities, run_model, SEED  # noqa: E402

# Display/report order: next-event group (A1-A3, B1) then full-suffix group
# (A4-A7, B2) - mirrors STATUS.md's own roster grouping, not alphabetical.
ROSTER_ORDER = [
    "process_transformer",
    "generative_lstm",
    "rlhgnn",
    "controlled_transformer_next",
    "sutran",
    "crtp_lstm",
    "lupin",
    "mlmme",
    "controlled_transformer_suffix",
]


def discover_models(dataset: str) -> list[str]:
    dataset_dir = REPO_ROOT / "results" / dataset
    available = {
        p.name
        for p in dataset_dir.iterdir()
        if p.is_dir() and (p / "embeddings_test.parquet").exists()
    }
    ordered = [m for m in ROSTER_ORDER if m in available]
    # Any model not in ROSTER_ORDER (shouldn't happen for the 9-model roster)
    # still gets included, just appended at the end rather than silently dropped.
    ordered += sorted(available - set(ordered))
    return ordered


def write_full_scale_summary(dataset: str, results: list[dict]) -> None:
    lines = [
        f"# Phase 6 full-scale — geometry descriptive results ({dataset})",
        "",
        f"{len(results)} model(s) analyzed: " + ", ".join(r["model"] for r in results) + ".",
        "",
        "Purely descriptive (PLAN.md's full-scale sub-phase scope) - not an H1-H3 "
        "hypothesis verdict and not cross-dataset statistical synthesis (Phase 7/10). "
        "Effective rank/participation ratio are computed in each model's own latent "
        "space independently (z_dim varies across the roster, spec Sec.14) - compare "
        "shape/pattern, never raw magnitude, across models.",
        "",
        "## Sanity",
        "",
    ]
    for r in results:
        status = "PASS" if not r["sanity_flags"] else "FLAGGED: " + "; ".join(r["sanity_flags"])
        lines.append(
            f"- **{r['model']}** (Family {r['family']}, objective={r['objective']}): "
            f"{status} ({r['elapsed_seconds']:.2f}s for {r['n_rows']} rows)"
        )

    header = "| metric | " + " | ".join(r["model"] for r in results) + " |"
    sep = "|---|" + "|".join(["---"] * len(results)) + "|"

    def row(label: str, fmt, getter):
        vals = []
        for r in results:
            v = getter(r)
            vals.append("NaN" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
        return f"| {label} | " + " | ".join(vals) + " |"

    lines += [
        "",
        "## Descriptive comparison",
        "",
        header,
        sep,
        row("z_dim", "{:d}", lambda r: r["diagnostics"]["z_dim"]),
        row("straightness (mean)", "{:.4f}", lambda r: r["trajectory"]["straightness_mean"]),
        row("path length (mean)", "{:.4f}", lambda r: r["trajectory"]["path_length_mean"]),
        row("effective rank", "{:.3f}", lambda r: r["diagnostics"]["effective_rank"]),
        row("participation ratio", "{:.3f}", lambda r: r["diagnostics"]["participation_ratio"]),
        row("terminal-state separability", "{:.4f}", lambda r: r["terminal"]["terminal_state_separability"]),
        row(
            "future rank corr. (edit dist.)",
            "{:.4f}",
            lambda r: r["future_equivalence"]["rank_correlation"]["edit_distance"],
        ),
        row(
            "future rank corr. (activity set)",
            "{:.4f}",
            lambda r: r["future_equivalence"]["rank_correlation"]["activity_set"],
        ),
        row(
            f"precision@{KNN_K} (edit dist.)",
            "{:.4f}",
            lambda r: r["future_equivalence"]["precision_at_k"]["edit_distance"],
        ),
        row(
            "trustworthiness (vs. edit-dist future)",
            "{:.4f}",
            lambda r: r["future_equivalence"]["trustworthiness_continuity_vs_edit_future"]["trustworthiness"],
        ),
        "",
    ]
    out_path = REPO_ROOT / "results" / "geometry_descriptive" / dataset / "full_scale" / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="helpdesk")
    parser.add_argument("models", nargs="*", help="Defaults to every model with completed embeddings.")
    args = parser.parse_args()

    models = args.models or discover_models(args.dataset)
    if not models:
        print(f"No completed (dataset, model) combinations found for dataset={args.dataset!r}.")
        return

    print(f"Phase 6 full-scale: {args.dataset} / {len(models)} models: {', '.join(models)}")
    acts_by_case, split_hash = load_raw_test_activities(args.dataset)
    rng = np.random.default_rng(SEED)

    results = []
    for model in models:
        r = run_model(args.dataset, model, acts_by_case, rng, run_label="full_scale")
        assert r["dataset_split_hash"] == split_hash, f"{model}: split hash mismatch vs. re-derived split"
        results.append(r)

    write_full_scale_summary(args.dataset, results)


if __name__ == "__main__":
    main()
