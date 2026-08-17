"""Validates geometry.future_equivalence (spec Sec.8.8) - future-distance
builders against hand-worked sequences, and the two "control for history
similarity" decisive tests against a fully hand-constructed toy scenario
with an analytically-known correlation."""
from __future__ import annotations

import numpy as np
import pytest

from geometry import future_equivalence as fe


def test_edit_distance_future_matches_dl_similarity():
    suffixes = [["a", "b"], ["a", "b"], ["x", "y"]]
    d = fe.edit_distance_future(suffixes)
    assert d[0, 1] == pytest.approx(0.0)  # identical suffixes
    assert d[0, 2] == pytest.approx(1.0)  # maximally dissimilar, same length
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_activity_set_distance_ignores_order():
    suffixes = [["a", "b"], ["b", "a"], ["a", "c"]]
    d = fe.activity_set_distance(suffixes)
    assert d[0, 1] == pytest.approx(0.0)  # same set, different order
    assert d[0, 2] == pytest.approx(1 - 1 / 3)  # Jaccard({a,b},{a,c}) = 1/3


def test_activity_set_distance_both_empty_is_zero():
    d = fe.activity_set_distance([[], []])
    assert d[0, 1] == pytest.approx(0.0)


def test_ngram_distance_is_order_sensitive():
    suffixes = [["a", "b", "c"], ["a", "b", "c"], ["c", "b", "a"]]
    d = fe.ngram_distance(suffixes, n=2)
    assert d[0, 1] == pytest.approx(0.0)  # identical bigrams
    assert d[0, 2] == pytest.approx(1.0)  # reversed order, no shared bigrams


def test_remaining_time_distance_normalized_by_max():
    d = fe.remaining_time_distance([0.0, 50.0, 100.0])
    assert d[0, 1] == pytest.approx(0.5)
    assert d[0, 2] == pytest.approx(1.0)
    assert d[1, 2] == pytest.approx(0.5)


def test_outcome_distance_categorical_and_null_handling():
    d = fe.outcome_distance(["y", "y", "n", None, None])
    assert d[0, 1] == pytest.approx(0.0)  # same label
    assert d[0, 2] == pytest.approx(1.0)  # different label
    assert d[3, 4] == pytest.approx(1.0)  # null is not a shared class, even with itself


def test_latent_distance_matrix_euclidean_and_cosine():
    Z = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    d = fe.latent_distance_matrix(Z, metric="euclidean")
    assert d[0, 1] == pytest.approx(5.0)
    assert d[0, 2] == pytest.approx(0.0)

    Z_dir = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    d_cos = fe.latent_distance_matrix(Z_dir, metric="cosine")
    assert d_cos[0, 1] == pytest.approx(1.0)  # orthogonal
    assert d_cos[0, 2] == pytest.approx(0.0)  # identical direction


def test_rank_correlation_perfect_and_inverse():
    d_latent = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    d_future_same_order = np.array([[0, 10, 20], [10, 0, 30], [20, 30, 0]], dtype=float)
    d_future_reversed = np.array([[0, 30, 20], [30, 0, 10], [20, 10, 0]], dtype=float)

    assert fe.rank_correlation(d_latent, d_future_same_order) == pytest.approx(1.0)
    assert fe.rank_correlation(d_latent, d_future_reversed) == pytest.approx(-1.0)


def test_precision_at_k_perfect_when_spaces_agree():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(10, 3))
    d = fe.latent_distance_matrix(Z)
    assert fe.precision_at_k(d, d, k=3) == pytest.approx(1.0)


def _hand_worked_history_scenario():
    """4 prefixes: (0,1) and (2,3) have similar observed histories; every
    other pair has dissimilar histories. d_latent and d_future are built so
    that, restricted to either the similar-history pairs or the dissimilar-
    history pairs, the relationship between the two is a perfect monotonic
    increase (Spearman rho = +1.0 exactly, by construction)."""
    d_history = np.array(
        [
            [0.0, 0.1, 0.9, 0.9],
            [0.1, 0.0, 0.9, 0.9],
            [0.9, 0.9, 0.0, 0.1],
            [0.9, 0.9, 0.1, 0.0],
        ]
    )
    # upper-triangle pair order (spec/_neighbors convention): (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    d_latent = np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 3.0, 4.0],
            [1.0, 3.0, 0.0, 2.0],
            [2.0, 4.0, 2.0, 0.0],
        ]
    )
    d_future = np.array(
        [
            [0.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 3.0, 4.0],
            [1.0, 3.0, 0.0, 3.0],
            [2.0, 4.0, 3.0, 0.0],
        ]
    )
    return d_latent, d_future, d_history


def test_dissimilar_history_retrieval_hand_worked():
    d_latent, d_future, d_history = _hand_worked_history_scenario()
    rho = fe.dissimilar_history_retrieval(d_latent, d_future, d_history, history_threshold=0.5)
    assert rho == pytest.approx(1.0)


def test_similar_history_divergence_hand_worked():
    d_latent, d_future, d_history = _hand_worked_history_scenario()
    rho = fe.similar_history_divergence(d_latent, d_future, d_history, history_threshold=0.5)
    assert rho == pytest.approx(1.0)
