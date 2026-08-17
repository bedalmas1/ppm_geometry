"""Representation-quality diagnostics (spec Sec.9).

Spec Sec.9 is explicit that trajectory metrics (geometry.trajectory) must
never be reported alone: "Straightness can be artificially high if
z_1~=z_2~=...~=z_T. ... A representation must not be labelled geometrically
superior merely because trajectories are straight." These diagnostics are
the required accompaniment, always computed on a *pooled* set of latent
vectors Z, shape (N, D) - e.g. every z_t across an entire test split for one
model - since collapse/triviality is a property of the whole learned space,
not any single trace.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

from geometry._neighbors import knn_sets, rank_matrix

_EPS = 1e-12


def embedding_variance(Z) -> dict:
    """Per-dimension variance and total variance (trace of the covariance
    matrix) of a pooled set of latent vectors. Near-zero total variance is
    the most direct symptom of representation collapse."""
    Z = np.asarray(Z, dtype=float)
    per_dim = Z.var(axis=0, ddof=0)
    return {"per_dimension": per_dim, "total": float(per_dim.sum())}


def covariance_spectrum(Z) -> np.ndarray:
    """Eigenvalues of the (D, D) covariance matrix of Z, descending order.
    Computed via SVD of the centered data (numerically stabler than eigh on
    an explicitly formed covariance matrix)."""
    Z = np.asarray(Z, dtype=float)
    centered = Z - Z.mean(axis=0, keepdims=True)
    n = len(centered)
    if n < 2:
        return np.zeros(Z.shape[1])
    s = np.linalg.svd(centered, compute_uv=False)
    return (s ** 2) / (n - 1)


def effective_rank(Z) -> float:
    """exp(Shannon entropy of the normalized covariance eigenvalues) - Roy &
    Vetterli's (2007) effective rank. 1.0 for a fully collapsed/rank-1
    representation, up to D for an isotropic one; a smooth alternative to a
    hard eigenvalue-threshold rank count."""
    eigenvalues = covariance_spectrum(Z)
    eigenvalues = eigenvalues[eigenvalues > _EPS]
    if len(eigenvalues) == 0:
        return 0.0
    p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def participation_ratio(Z) -> float:
    """(sum(lambda))^2 / sum(lambda^2) over the covariance spectrum - a
    standard linear intrinsic-dimensionality estimator. Deliberately not a
    nonlinear manifold estimator (e.g. TwoNN): a documented scope decision,
    sufficient as a first-pass diagnostic; revisit only if a later phase
    needs a nonlinear estimate instead."""
    eigenvalues = covariance_spectrum(Z)
    s1 = eigenvalues.sum()
    s2 = np.square(eigenvalues).sum()
    if s2 <= _EPS:
        return 0.0
    return float((s1 ** 2) / s2)


def pairwise_distance_summary(Z, metric: str = "euclidean") -> dict:
    """Summary statistics of the condensed pairwise-distance distribution
    (spec Sec.9's "pairwise distance distribution"). A collapsed or
    near-collapsed space shows a distribution concentrated near 0."""
    Z = np.asarray(Z, dtype=float)
    if len(Z) < 2:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "median": float("nan")}
    d = pdist(Z, metric=metric)
    return {
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "median": float(np.median(d)),
    }


def terminal_state_separability(Z_terminal, labels) -> float:
    """Silhouette score of terminal-state vectors grouped by label (e.g.
    outcome or terminal-variant) - a standard, bounded [-1, 1] measure of
    cluster separation. NaN if fewer than 2 distinct labels, or any label
    has fewer than 2 members (silhouette is undefined then), rather than
    raising or silently returning a misleading 0."""
    Z_terminal = np.asarray(Z_terminal, dtype=float)
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2):
        return float("nan")
    return float(silhouette_score(Z_terminal, labels))


def trustworthiness(d_true: np.ndarray, d_embedded: np.ndarray, k: int) -> float:
    """How much the "embedded" space's k-NN structure introduces false
    neighbors that are not actually close in the "true" space (Venna &
    Kaski 2001). 1.0 = no false neighbors (perfect); lower is worse.

    Implemented directly on two precomputed (N, N) distance matrices, rather
    than sklearn.manifold.trustworthiness's coordinate-only API, since this
    project needs it between two independently-defined *distances* (e.g.
    latent d_Z vs. future-suffix d_F, spec Sec.8.8) that are never both
    literal coordinate embeddings of the same space.

    NaN if there are too few points to form k neighbors with margin.
    O(N^2) in points and neighbors examined - fine at synthetic-validation
    and per-dataset-per-model scale; a real-scale (many thousands of
    prefixes) application should subsample or vectorize further, an
    optimization deliberately deferred rather than premature here.
    """
    d_true = np.asarray(d_true, dtype=float)
    d_embedded = np.asarray(d_embedded, dtype=float)
    n = d_true.shape[0]
    if n <= k + 1:
        return float("nan")

    rank_true = rank_matrix(d_true)
    knn_embedded = knn_sets(d_embedded, k)
    knn_true = knn_sets(d_true, k)

    penalty = 0.0
    for i in range(n):
        for j in knn_embedded[i] - knn_true[i]:
            penalty += rank_true[i, j] - k

    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - norm * penalty)


def continuity(d_true: np.ndarray, d_embedded: np.ndarray, k: int) -> float:
    """Symmetric counterpart to trustworthiness: penalizes true neighbors
    that the embedded space fails to keep close, rather than false
    neighbors it introduces. continuity(A, B, k) == trustworthiness(B, A, k)
    by definition."""
    return trustworthiness(d_embedded, d_true, k)
