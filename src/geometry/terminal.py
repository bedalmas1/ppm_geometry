"""Progress toward terminal regions (spec Sec.8.6).

Two prototype definitions, per the spec's explicit instruction not to assume
Euclidean prototypes are always valid: a Euclidean class centroid (c_y =
E[z_T | Y=y]) and a k-NN neighborhood/distribution-based alternative (mean
distance to the k nearest same-class terminal points) that survives a
multimodal terminal region a single centroid would average into a
meaningless midpoint.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def terminal_centroids(z_terminal, labels) -> dict:
    """c_y = mean of z_T over every terminal vector with Y=y, for each
    distinct label y present in `labels`."""
    z_terminal = np.asarray(z_terminal, dtype=float)
    labels = np.asarray(labels)
    return {y: z_terminal[labels == y].mean(axis=0) for y in np.unique(labels)}


def distance_to_prototype(z, prototype, metric: str = "euclidean") -> np.ndarray:
    """d_y(t) = d(z_t, c_y) (spec Sec.8.6), for every t in trajectory z.
    Shape (T,)."""
    z = np.asarray(z, dtype=float)
    prototype = np.asarray(prototype, dtype=float)
    if metric == "euclidean":
        return np.linalg.norm(z - prototype, axis=1)
    if metric == "cosine":
        z_norm = np.linalg.norm(z, axis=1)
        p_norm = np.linalg.norm(prototype)
        denom = z_norm * p_norm
        cos_sim = np.divide(
            z @ prototype, denom, out=np.zeros_like(z_norm), where=denom > 1e-12
        )
        return 1.0 - cos_sim
    raise ValueError(f"unknown metric {metric!r}")


def knn_distance_to_class(z_query, z_terminal, labels, target_label, k: int) -> np.ndarray:
    """Neighborhood/distribution-based alternative to a single centroid: for
    each query vector, the mean distance to its k nearest terminal vectors
    labelled `target_label`. Robust to a multimodal terminal region (e.g.
    two distinct clusters sharing one outcome label) that a Euclidean
    centroid would misrepresent as a single midpoint no real case is near."""
    z_query = np.asarray(z_query, dtype=float)
    labels = np.asarray(labels)
    class_points = np.asarray(z_terminal, dtype=float)[labels == target_label]
    if len(class_points) == 0:
        return np.full(len(z_query), np.nan)
    k = min(k, len(class_points))
    dists = np.linalg.norm(z_query[:, None, :] - class_points[None, :, :], axis=2)
    dists.sort(axis=1)
    return dists[:, :k].mean(axis=1)


def progress_trend(distances) -> float:
    """Spearman rank correlation between prefix index t and distance-to-
    prototype d_y(t). Negative -> distance shrinks as the case progresses,
    the qualitative pattern spec Sec.8.6 asks whether models exhibit; this
    function only measures the trend, it does not itself claim the pattern
    is causal or universally desirable (spec Sec.18)."""
    distances = np.asarray(distances, dtype=float)
    if len(distances) < 2 or np.all(distances == distances[0]):
        return float("nan")
    t = np.arange(len(distances))
    rho, _ = spearmanr(t, distances)
    return float(rho)
