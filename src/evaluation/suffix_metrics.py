"""Suffix-prediction similarity metrics.

Normalized Damerau-Levenshtein (DL) similarity is the standard metric for
comparing predicted and ground-truth activity suffixes in the PPM literature
(used by SuTraN, CRTP-LSTM, Camargo et al., and others) and is also the
distance the research spec names for the Phase 8 future-equivalence study
(spec Sec.8.8) - implemented once here, reused by both.
"""
from __future__ import annotations

from typing import Sequence


def damerau_levenshtein_distance(a: Sequence, b: Sequence) -> int:
    """Edit distance allowing insertion, deletion, substitution, and
    transposition of adjacent elements. Works on any sequence of hashable
    tokens (strings, ints, ...), not just characters."""
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    da: dict = {}
    max_dist = len_a + len_b
    d = [[0] * (len_b + 2) for _ in range(len_a + 2)]
    d[0][0] = max_dist
    for i in range(0, len_a + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(0, len_b + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    for i in range(1, len_a + 1):
        db = 0
        for j in range(1, len_b + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            if a[i - 1] == b[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1
            d[i + 1][j + 1] = min(
                d[i][j] + cost,  # substitution (or match)
                d[i + 1][j] + 1,  # insertion
                d[i][j + 1] + 1,  # deletion
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),  # transposition
            )
        da[a[i - 1]] = i

    return d[len_a + 1][len_b + 1]


def normalized_dl_similarity(predicted: Sequence, ground_truth: Sequence) -> float:
    """1 - DL(predicted, ground_truth) / max(len(predicted), len(ground_truth)).

    1.0 = identical sequences, 0.0 = maximally dissimilar. Both empty ->
    defined as 1.0 (identical, matches SuTraN's own convention)."""
    if len(predicted) == 0 and len(ground_truth) == 0:
        return 1.0
    dist = damerau_levenshtein_distance(predicted, ground_truth)
    return 1.0 - dist / max(len(predicted), len(ground_truth))
