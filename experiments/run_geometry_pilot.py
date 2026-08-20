#!/usr/bin/env python
"""Phase 6 pilot sub-phase (PLAN.md's mandatory gate before scaling): run the
geometry package (src/geometry/) against real, already-extracted Phase 4
embeddings for one dataset x two models, and check the results behave
sanely before trusting any trajectory metric at full scale.

Per PLAN.md's recommendation, defaults to the pilot pair PLAN.md/STATUS.md
name explicitly: B1 controlled_transformer_next vs. B2
controlled_transformer_suffix on Helpdesk - both already trained (Phase 3)
and extracted (Phase 4), so this script does no training/extraction of its
own, only analysis of results/<dataset>/<model>/embeddings_test.parquet.

Usage:
    uv run python experiments/run_geometry_pilot.py [dataset] [model1] [model2]

No deep-learning framework needed (numpy/pandas/scipy/sklearn only, all
already project dependencies via src/geometry) - the geometry-analysis step
itself is cheap; Phase 1's compute-budget concern was about training/
extraction, not this.

Ground-truth "future" per prefix (used by geometry.future_equivalence) is
recomputed here directly from the raw event log/split (case activities after
index k), independent of whichever objective the model itself was trained
on - this lets B1 (next-event, no stored suffix) and B2 (full-suffix,
already stores true_suffix) be evaluated identically.

Outputs one results/geometry_descriptive/<dataset>/pilot/<model>/*.json per
model plus a comparison results/geometry_descriptive/<dataset>/pilot/summary.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.loaders import load_dataset  # noqa: E402
from data.prefixes import normalize_activity  # noqa: E402
from data.schema import ACTIVITY, CASE_ID  # noqa: E402
from data.splits import apply_split, compute_split  # noqa: E402
from geometry import branching, diagnostics, terminal, trajectory  # noqa: E402
from geometry.future_equivalence import (  # noqa: E402
    activity_set_distance,
    continuity,
    dissimilar_history_retrieval,
    edit_distance_future,
    latent_distance_matrix,
    ngram_distance,
    precision_at_k,
    rank_correlation,
    remaining_time_distance,
    similar_history_divergence,
    trustworthiness,
)

N_BINS = 10
KNN_K = 10
SUBSAMPLE_SIZE = 500  # future_equivalence/trustworthiness are O(n^2); see
# STATUS.md open question re: subsampling before running at full-cache scale.
SEED = 42


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def software_versions() -> dict:
    versions = {}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn"):
        try:
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    return versions


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays and NaN/inf to plain,
    strictly-valid JSON (None for non-finite floats) - these outputs are
    meant to be read by later phases/paper-figure code, not just humans."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def load_raw_test_activities(dataset: str) -> tuple[dict, str]:
    """case_id -> normalized activity list, for the dataset's test split -
    used as model-independent ground truth (future suffixes, terminal
    activity) rather than anything derived from a specific model's own
    training objective."""
    dataset_config = yaml.safe_load((REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml").read_text(encoding="utf-8"))
    df = load_dataset(REPO_ROOT / dataset_config["raw_path"], dataset_config["source"]["format"], dataset_config.get("date_filter"))
    split_cfg = dataset_config["split"]
    split = compute_split(df, split_cfg["train_frac"], split_cfg["val_frac"], split_cfg["test_frac"])
    parts = apply_split(df, split)
    test_df = parts["test"]
    acts_by_case = {
        case_id: [normalize_activity(a) for a in group[ACTIVITY].tolist()]
        for case_id, group in test_df.groupby(CASE_ID, sort=False)
    }
    return acts_by_case, split.split_hash


def load_model_embeddings(dataset: str, model: str) -> pd.DataFrame:
    result_dir = REPO_ROOT / "results" / dataset / model
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    emb_manifest = json.loads((result_dir / "embeddings_manifest.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(result_dir / "embeddings_test.parquet")
    df["z"] = df["z"].apply(lambda v: np.asarray(v, dtype=float))
    return df, manifest, emb_manifest


def build_case_trajectories(df: pd.DataFrame) -> dict:
    """case_id -> (T, D) latent trajectory, k-sorted."""
    trajectories = {}
    for case_id, group in df.groupby("case_id", sort=False):
        group = group.sort_values("k")
        trajectories[case_id] = np.stack(group["z"].to_numpy())
    return trajectories


def trajectory_summary(trajectories: dict) -> dict:
    straightness_vals, path_len_vals = [], []
    velocity_vals, curvature_vals, smoothness_vals = [], [], []
    for z in trajectories.values():
        straightness_vals.append(trajectory.straightness(z))
        path_len_vals.append(trajectory.path_length(z))
        v = trajectory.velocity(z)
        c = trajectory.curvature(z)
        s = trajectory.smoothness(z)
        if len(v):
            velocity_vals.append(np.nanmean(v))
        if len(c):
            curvature_vals.append(np.nanmean(c))
        if len(s):
            smoothness_vals.append(np.nanmean(s))

    def _summ(vals, name):
        arr = np.asarray(vals, dtype=float)
        n_nan = int(np.isnan(arr).sum())
        return {
            f"{name}_mean": float(np.nanmean(arr)) if n_nan < len(arr) else float("nan"),
            f"{name}_median": float(np.nanmedian(arr)) if n_nan < len(arr) else float("nan"),
            f"{name}_n_nan": n_nan,
            f"{name}_n_total": len(arr),
        }

    out = {}
    out.update(_summ(straightness_vals, "straightness"))
    out.update(_summ(path_len_vals, "path_length"))
    out.update(_summ(velocity_vals, "velocity"))
    out.update(_summ(curvature_vals, "curvature"))
    out.update(_summ(smoothness_vals, "smoothness"))
    out["n_traces"] = len(trajectories)
    return out


def diagnostics_summary(Z: np.ndarray) -> dict:
    var = diagnostics.embedding_variance(Z)
    spectrum = diagnostics.covariance_spectrum(Z)
    total = spectrum.sum()
    top_k = min(10, len(spectrum))
    return {
        "total_variance": var["total"],
        "effective_rank": diagnostics.effective_rank(Z),
        "participation_ratio": diagnostics.participation_ratio(Z),
        "z_dim": Z.shape[1],
        "n_pooled_vectors": Z.shape[0],
        "top_eigenvalues": spectrum[:top_k].tolist(),
        "top_k_variance_fraction": float(spectrum[:top_k].sum() / total) if total > 0 else float("nan"),
        "pairwise_distance": diagnostics.pairwise_distance_summary(Z),
    }


def terminal_and_branching(df: pd.DataFrame, trajectories: dict, acts_by_case: dict) -> tuple[dict, dict]:
    case_ids = list(trajectories.keys())
    terminal_labels = np.array([acts_by_case[c][-1] for c in case_ids])
    terminal_z = np.stack([trajectories[c][-1] for c in case_ids])

    term_out = {
        "terminal_state_separability": diagnostics.terminal_state_separability(terminal_z, terminal_labels),
        "n_distinct_terminal_activities": int(len(np.unique(terminal_labels))),
        "terminal_activity_counts": {str(a): int(c) for a, c in zip(*np.unique(terminal_labels, return_counts=True))},
    }

    row_labels = df["case_id"].map({c: acts_by_case[c][-1] for c in case_ids}).to_numpy()
    s = branching.normalized_progress(df["k"].to_numpy(), df["case_length"].to_numpy())
    curve = branching.branch_separation_curve(s, row_labels, np.stack(df["z"].to_numpy()), n_bins=N_BINS)
    branch_out = {"separation_curve": curve}
    return term_out, branch_out


def future_equivalence_summary(df: pd.DataFrame, acts_by_case: dict, rng: np.random.Generator) -> dict:
    n = len(df)
    sample_size = min(SUBSAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_size, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)

    Z = np.stack(sub["z"].to_numpy())
    futures = [acts_by_case[row.case_id][row.k + 1 :] for row in sub.itertuples(index=False)]
    histories = [acts_by_case[row.case_id][: row.k + 1] for row in sub.itertuples(index=False)]
    remaining_times = sub["remaining_time_seconds"].to_numpy()

    d_Z = latent_distance_matrix(Z)
    d_history = edit_distance_future(histories)
    d_F_edit = edit_distance_future(futures)
    d_F_activity_set = activity_set_distance(futures)
    d_F_ngram = ngram_distance(futures, n=2)
    d_F_time = remaining_time_distance(remaining_times)

    threshold = float(np.median(d_history[np.triu_indices(len(d_history), k=1)]))

    return {
        "sample_size": sample_size,
        "history_threshold_median": threshold,
        "rank_correlation": {
            "edit_distance": rank_correlation(d_Z, d_F_edit),
            "activity_set": rank_correlation(d_Z, d_F_activity_set),
            "ngram2": rank_correlation(d_Z, d_F_ngram),
            "remaining_time": rank_correlation(d_Z, d_F_time),
        },
        "precision_at_k": {
            "edit_distance": precision_at_k(d_Z, d_F_edit, KNN_K),
            "activity_set": precision_at_k(d_Z, d_F_activity_set, KNN_K),
        },
        "trustworthiness_continuity_vs_edit_future": {
            "trustworthiness": trustworthiness(d_F_edit, d_Z, KNN_K),
            "continuity": continuity(d_F_edit, d_Z, KNN_K),
        },
        "decisive_tests_edit_distance": {
            "dissimilar_history_retrieval": dissimilar_history_retrieval(d_Z, d_F_edit, d_history, threshold),
            "similar_history_divergence": similar_history_divergence(d_Z, d_F_edit, d_history, threshold),
        },
    }


def sanity_checks(diag: dict, traj: dict) -> list[str]:
    flags = []
    if diag["total_variance"] <= 1e-9:
        flags.append("COLLAPSE: pooled embedding total variance is ~0.")
    if diag["effective_rank"] < 1.5:
        flags.append(f"COLLAPSE-LIKE: effective_rank={diag['effective_rank']:.3f} is close to 1.")
    if traj["straightness_n_nan"] == traj["straightness_n_total"]:
        flags.append("DEGENERATE: every trace's straightness is NaN (all traces length<2 or collapsed).")
    if traj["straightness_n_nan"] > 0.5 * traj["straightness_n_total"]:
        flags.append(
            f"WARNING: {traj['straightness_n_nan']}/{traj['straightness_n_total']} traces have NaN straightness."
        )
    return flags


def run_model(
    dataset: str, model: str, acts_by_case: dict, rng: np.random.Generator, run_label: str = "pilot"
) -> dict:
    t0 = time.perf_counter()
    df, manifest, emb_manifest = load_model_embeddings(dataset, model)

    trajectories = build_case_trajectories(df)
    Z_pooled = np.stack(df["z"].to_numpy())

    traj = trajectory_summary(trajectories)
    diag = diagnostics_summary(Z_pooled)
    term, branch = terminal_and_branching(df, trajectories, acts_by_case)
    future_eq = future_equivalence_summary(df, acts_by_case, rng)
    flags = sanity_checks(diag, traj)

    elapsed = time.perf_counter() - t0

    result = {
        "dataset": dataset,
        "model": model,
        "family": manifest["model_config"]["family"],
        "objective": manifest["model_config"]["objective"],
        "dataset_split_hash": emb_manifest["dataset_split_hash"],
        "n_rows": len(df),
        "n_traces": len(trajectories),
        "elapsed_seconds": elapsed,
        "trajectory": traj,
        "diagnostics": diag,
        "terminal": term,
        "branching": branch,
        "future_equivalence": future_eq,
        "sanity_flags": flags,
    }

    out_dir = REPO_ROOT / "results" / "geometry_descriptive" / dataset / run_label / model
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "geometry_report.json", result)

    provenance = {
        "dataset": dataset,
        "model": model,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "software_versions": software_versions(),
        "source_embeddings_manifest": emb_manifest,
        "subsample_size": SUBSAMPLE_SIZE,
        "subsample_seed": SEED,
        "elapsed_seconds": elapsed,
    }
    _write_json(out_dir / "provenance.json", provenance)

    print(f"[{dataset}/{model}] {len(df)} rows / {len(trajectories)} traces analyzed in {elapsed:.2f}s")
    for flag in flags:
        print(f"[{dataset}/{model}]   ! {flag}")
    if not flags:
        print(f"[{dataset}/{model}]   sanity checks passed (no collapse/degeneracy flags)")

    return result


def write_summary(dataset: str, results: list[dict]) -> None:
    a, b = results
    lines = [
        f"# Phase 6 pilot — geometry descriptive results ({dataset})",
        "",
        f"Pilot pair: **{a['model']}** (Family {a['family']}, objective={a['objective']}) vs. "
        f"**{b['model']}** (Family {b['family']}, objective={b['objective']}).",
        "",
        "This is the mandatory pilot sub-phase gate (PLAN.md Phase 6) — validates the geometry "
        "package behaves sanely on real (non-synthetic) embeddings before scaling to the full "
        "9-model x 5-dataset matrix. Not a full H1-H3 verdict (that's the full-scale sub-phase).",
        "",
        "## Sanity",
        "",
    ]
    for r in results:
        status = "PASS" if not r["sanity_flags"] else "FLAGGED: " + "; ".join(r["sanity_flags"])
        lines.append(f"- **{r['model']}**: {status} ({r['elapsed_seconds']:.2f}s for {r['n_rows']} rows)")

    lines += [
        "",
        "## Descriptive comparison",
        "",
        "| metric | " + a["model"] + " | " + b["model"] + " |",
        "|---|---|---|",
        f"| straightness (mean) | {a['trajectory']['straightness_mean']:.4f} | {b['trajectory']['straightness_mean']:.4f} |",
        f"| path length (mean) | {a['trajectory']['path_length_mean']:.4f} | {b['trajectory']['path_length_mean']:.4f} |",
        f"| effective rank | {a['diagnostics']['effective_rank']:.3f} | {b['diagnostics']['effective_rank']:.3f} |",
        f"| participation ratio | {a['diagnostics']['participation_ratio']:.3f} | {b['diagnostics']['participation_ratio']:.3f} |",
        f"| terminal-state separability | {a['terminal']['terminal_state_separability']:.4f} | {b['terminal']['terminal_state_separability']:.4f} |",
        f"| future rank corr. (edit dist.) | {a['future_equivalence']['rank_correlation']['edit_distance']:.4f} | {b['future_equivalence']['rank_correlation']['edit_distance']:.4f} |",
        f"| precision@{KNN_K} (edit dist.) | {a['future_equivalence']['precision_at_k']['edit_distance']:.4f} | {b['future_equivalence']['precision_at_k']['edit_distance']:.4f} |",
        "",
        "Effective rank and participation ratio are computed on each model's own latent space "
        "independently (dimensions differ across the roster, spec Sec.14) — compare shape/pattern, "
        "never raw magnitude, across models with different z_dim.",
        "",
    ]
    out_path = REPO_ROOT / "results" / "geometry_descriptive" / dataset / "pilot" / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="helpdesk")
    parser.add_argument("model_a", nargs="?", default="controlled_transformer_next")
    parser.add_argument("model_b", nargs="?", default="controlled_transformer_suffix")
    args = parser.parse_args()

    print(f"Phase 6 pilot: {args.dataset} / {args.model_a} vs {args.model_b}")
    acts_by_case, split_hash = load_raw_test_activities(args.dataset)
    rng = np.random.default_rng(SEED)

    results = []
    for model in (args.model_a, args.model_b):
        r = run_model(args.dataset, model, acts_by_case, rng)
        assert r["dataset_split_hash"] == split_hash, f"{model}: split hash mismatch vs. re-derived split"
        results.append(r)

    write_summary(args.dataset, results)


if __name__ == "__main__":
    main()
