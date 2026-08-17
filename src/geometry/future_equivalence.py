"""Future-equivalence neighborhood quality (spec Sec.8.8).

Compares latent distance d_Z(z_i, z_j) between prefixes against a notion of
"future similarity" d_F(tau_i, tau_j) defined independently of the latent
space, purely from *observable* suffixes - never from a model's own
predictions ("learned suffix distance" is explicitly named as a secondary,
optional measure only, spec Sec.8.8).

Every d_F builder below returns a (N, N) distance matrix over N prefixes,
indexed consistently with the caller's own d_Z matrix, so the two can be
compared entry-by-entry (rank_correlation, precision_at_k) or via the
trustworthiness/continuity primitives in geometry.diagnostics.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from evaluation.suffix_metrics import normalized_dl_similarity
from geometry._neighbors import knn_sets, upper_triangle
from geometry.diagnostics import continuity, trustworthiness

__all__ = [
    "latent_distance_matrix",
    "edit_distance_future",
    "activity_set_distance",
    "ngram_distance",
    "remaining_time_distance",
    "outcome_distance",
    "rank_correlation",
    "precision_at_k",
    "trustworthiness",
    "continuity",
    "dissimilar_history_retrieval",
    "similar_history_divergence",
]


def _pairwise(n: int, distance_fn) -> np.ndarray:
    """Fill a symmetric (n, n) matrix from a pairwise scalar distance_fn(i,
    j), zero diagonal. Shared by every d_F builder below - none of the
    per-pair suffix/activity-set/n-gram distances have a vectorized
    closed form, so this is a plain O(n^2) double loop, acceptable at
    per-model/per-dataset analysis scale."""
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = distance_fn(i, j)
            d[i, j] = d[j, i] = value
    return d


def latent_distance_matrix(Z, metric: str = "euclidean") -> np.ndarray:
    """d_Z: pairwise distance matrix over a set of latent prefix vectors."""
    Z = np.asarray(Z, dtype=float)
    if metric == "euclidean":
        diff = Z[:, None, :] - Z[None, :, :]
        return np.linalg.norm(diff, axis=2)
    if metric == "cosine":
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        normed = Z / np.where(norms > 1e-12, norms, 1.0)
        sim = normed @ normed.T
        return 1.0 - np.clip(sim, -1.0, 1.0)
    raise ValueError(f"unknown metric {metric!r}")


def edit_distance_future(suffixes: Sequence[Sequence]) -> np.ndarray:
    """d_F via normalized Damerau-Levenshtein distance between observed
    future suffixes: 1 - normalized_dl_similarity(suffix_i, suffix_j)."""
    n = len(suffixes)
    return _pairwise(n, lambda i, j: 1.0 - normalized_dl_similarity(suffixes[i], suffixes[j]))


def activity_set_distance(suffixes: Sequence[Sequence]) -> np.ndarray:
    """d_F via 1 - Jaccard similarity of the *set* of activities appearing
    in each future suffix (ignores order/repetition, unlike edit distance).
    Two empty suffixes are defined as identical (distance 0)."""
    sets = [set(s) for s in suffixes]

    def _dist(i: int, j: int) -> float:
        a, b = sets[i], sets[j]
        if not a and not b:
            return 0.0
        return 1.0 - len(a & b) / len(a | b)

    return _pairwise(len(suffixes), _dist)


def ngram_distance(suffixes: Sequence[Sequence], n: int = 2) -> np.ndarray:
    """d_F via 1 - Jaccard similarity of the set of length-n contiguous
    subsequences (n-grams) of each suffix - sensitive to local order,
    unlike activity_set_distance, but more tolerant of a few
    insertions/deletions than edit_distance_future. A suffix shorter than
    n contributes its own full sequence as a single "n-gram" (its longest
    available fragment) rather than an empty set."""

    def _ngrams(seq: Sequence) -> set:
        seq = tuple(seq)
        if len(seq) < n:
            return {seq} if seq else set()
        return {seq[i : i + n] for i in range(len(seq) - n + 1)}

    grams = [_ngrams(s) for s in suffixes]

    def _dist(i: int, j: int) -> float:
        a, b = grams[i], grams[j]
        if not a and not b:
            return 0.0
        return 1.0 - len(a & b) / len(a | b)

    return _pairwise(len(suffixes), _dist)


def remaining_time_distance(remaining_times: Sequence[float]) -> np.ndarray:
    """d_F via absolute difference in remaining time, normalized by the
    largest observed remaining time in this set so the result stays in
    [0, 1] regardless of the dataset's raw time unit."""
    times = np.asarray(remaining_times, dtype=float)
    scale = np.max(np.abs(times)) if len(times) and np.max(np.abs(times)) > 1e-12 else 1.0
    diff = np.abs(times[:, None] - times[None, :]) / scale
    return diff


def outcome_distance(outcomes: Sequence) -> np.ndarray:
    """d_F via a categorical 0/1 distance: 0 if two prefixes share the same
    non-null outcome label, 1 otherwise. `None`/missing outcomes are always
    distance 1 from everything, including each other - "missing" is not a
    shared class, never silently dropped or guessed."""
    outcomes = list(outcomes)

    def _dist(i: int, j: int) -> float:
        a, b = outcomes[i], outcomes[j]
        if a is None or b is None:
            return 1.0
        return 0.0 if a == b else 1.0

    return _pairwise(len(outcomes), _dist)


def rank_correlation(d_latent: np.ndarray, d_future: np.ndarray) -> float:
    """Spearman rank correlation between d_Z and d_F over every unordered
    prefix pair. Positive -> prefixes close in latent space tend to have
    similar futures; this is purely descriptive of association, not a
    causal claim (spec Sec.18)."""
    pairs_latent = upper_triangle(d_latent)
    pairs_future = upper_triangle(d_future)
    if len(pairs_latent) < 2 or np.all(pairs_latent == pairs_latent[0]):
        return float("nan")
    rho, _ = spearmanr(pairs_latent, pairs_future)
    return float(rho)


def precision_at_k(d_latent: np.ndarray, d_future: np.ndarray, k: int) -> float:
    """Fraction of each prefix's k latent-nearest-neighbors that are also
    among its k future-nearest-neighbors (observable-suffix ground truth),
    averaged over all prefixes - "nearest-neighbor retrieval quality" per
    spec Sec.8.8."""
    knn_latent = knn_sets(d_latent, k)
    knn_future = knn_sets(d_future, k)
    n = d_latent.shape[0]
    if n == 0 or k == 0:
        return float("nan")
    precisions = [len(knn_latent[i] & knn_future[i]) / k for i in range(n)]
    return float(np.mean(precisions))


def dissimilar_history_retrieval(
    d_latent: np.ndarray, d_future: np.ndarray, d_history: np.ndarray, history_threshold: float
) -> float:
    """Spec Sec.8.8's first decisive test: "Among prefixes with dissimilar
    histories, does latent proximity retrieve cases with similar futures?"

    Restricts rank_correlation(d_Z, d_F) to pairs whose observed-history
    distance exceeds `history_threshold` (i.e. genuinely dissimilar
    prefixes) - if latent proximity is only ever explained by shared
    history, this correlation collapses toward 0 once history similarity is
    controlled for."""
    mask = upper_triangle(d_history) > history_threshold
    latent_pairs = upper_triangle(d_latent)[mask]
    future_pairs = upper_triangle(d_future)[mask]
    if len(latent_pairs) < 2 or np.all(latent_pairs == latent_pairs[0]):
        return float("nan")
    rho, _ = spearmanr(latent_pairs, future_pairs)
    return float(rho)


def similar_history_divergence(
    d_latent: np.ndarray, d_future: np.ndarray, d_history: np.ndarray, history_threshold: float
) -> float:
    """Spec Sec.8.8's second decisive test: "Among prefixes with similar
    histories, does latent distance increase when their futures diverge?"

    Restricts rank_correlation(d_Z, d_F) to pairs whose observed-history
    distance is at most `history_threshold` (near-identical prefixes) -
    a positive correlation here means the model keeps histories that will
    diverge apart in latent space even while their observed pasts still
    look alike, beyond what shared history alone would predict."""
    mask = upper_triangle(d_history) <= history_threshold
    latent_pairs = upper_triangle(d_latent)[mask]
    future_pairs = upper_triangle(d_future)[mask]
    if len(latent_pairs) < 2 or np.all(latent_pairs == latent_pairs[0]):
        return float("nan")
    rho, _ = spearmanr(latent_pairs, future_pairs)
    return float(rho)
