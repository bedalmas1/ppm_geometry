"""Branch separation over normalized prefix progress (spec Sec.8.7).

Groups are e.g. cases heading to different outcomes/future variants.
Progress is normalized to s = t / (case_length - 1) in [0, 1] so cases of
different lengths align on one shared axis, then binned so groups can be
compared at matched progress levels rather than matched raw prefix index.
"""
from __future__ import annotations

import numpy as np


def normalized_progress(t, case_length) -> np.ndarray:
    """s = t / (case_length - 1), in [0, 1]. A length-1 case (no progress
    possible at all) maps to s=0 rather than dividing by zero."""
    t = np.asarray(t, dtype=float)
    case_length = np.asarray(case_length, dtype=float)
    denom = np.where(case_length > 1, case_length - 1, 1.0)
    return t / denom


def assign_progress_bins(s, n_bins: int) -> np.ndarray:
    """Bin index in [0, n_bins - 1] for each normalized progress value."""
    s = np.asarray(s, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)


def group_separation(group_vectors) -> float:
    """Separation among >=2 groups of latent vectors at one progress level:
    the ratio of between-group scatter to pooled within-group scatter - a
    multi-group generalization of Fisher's separation criterion. NaN if
    fewer than 2 non-empty groups are present, or within-group scatter is 0
    (degenerate/collapsed groups - separation is then undefined, not
    infinite)."""
    groups = [np.asarray(g, dtype=float) for g in group_vectors if len(g) > 0]
    if len(groups) < 2:
        return float("nan")

    overall_mean = np.concatenate(groups, axis=0).mean(axis=0)
    between = 0.0
    within = 0.0
    for g in groups:
        group_mean = g.mean(axis=0)
        between += len(g) * np.sum((group_mean - overall_mean) ** 2)
        within += np.sum((g - group_mean) ** 2)

    if within <= 1e-12:
        return float("nan")
    return float(between / within)


def branch_separation_curve(s, group_labels, vectors, n_bins: int = 10) -> dict:
    """Separation statistic per normalized-progress bin:
    {bin_center: separation}. A bin where fewer than 2 groups have any
    points is omitted entirely (not zero-filled), so a caller can see where
    a comparison was even possible before reading anything off the curve -
    spec Sec.8.7 requires defining a bifurcation/predictive-horizon time
    only "if supported by robust statistical analysis", never assumed from
    a sparse or partially-undefined curve.
    """
    s = np.asarray(s, dtype=float)
    group_labels = np.asarray(group_labels)
    vectors = np.asarray(vectors, dtype=float)
    bins = assign_progress_bins(s, n_bins)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins

    curve = {}
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        groups = [vectors[mask & (group_labels == g)] for g in np.unique(group_labels)]
        sep = group_separation(groups)
        if not np.isnan(sep):
            curve[float(bin_centers[b])] = sep
    return curve
