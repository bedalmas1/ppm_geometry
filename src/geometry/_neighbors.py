"""Shared k-nearest-neighbor primitives over precomputed distance matrices.

Used by geometry.diagnostics (trustworthiness/continuity, spec Sec.9) and
geometry.future_equivalence (precision@k, spec Sec.8.8) - both compare two
independently-defined notions of distance on the same set of points (e.g.
latent distance vs. future-suffix distance), so both need "who are point i's
k nearest neighbors, and at what rank" from an arbitrary (N, N) distance
matrix, not just from raw coordinates.
"""
from __future__ import annotations

import numpy as np


def knn_sets(d: np.ndarray, k: int) -> list[set[int]]:
    """N_k(i): the indices of the k nearest neighbors of i (excluding i
    itself), ascending by distance, ties broken by index order."""
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    order = np.argsort(d, axis=1, kind="stable")
    neighbor_sets = []
    for i in range(n):
        others = [j for j in order[i] if j != i][:k]
        neighbor_sets.append(set(others))
    return neighbor_sets


def rank_matrix(d: np.ndarray) -> np.ndarray:
    """rank[i, j] = 1-based rank of j among the neighbors of i (nearest
    neighbor excluding self = rank 1). rank[i, i] is left at 0 and must
    never be read (i is never its own neighbor)."""
    d = np.asarray(d, dtype=float)
    n = d.shape[0]
    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.zeros((n, n), dtype=int)
    for i in range(n):
        others = [j for j in order[i] if j != i]
        for rank, j in enumerate(others, start=1):
            ranks[i, j] = rank
    return ranks


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Flattened i<j entries of a square matrix, e.g. to correlate two
    pairwise-distance matrices defined over the identical point ordering."""
    mat = np.asarray(mat)
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]
