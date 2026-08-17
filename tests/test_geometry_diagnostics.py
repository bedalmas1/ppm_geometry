"""Validates geometry.diagnostics (spec Sec.9) - the collapse/triviality
guards that must accompany every trajectory metric."""
from __future__ import annotations

import numpy as np
import pytest

from geometry import diagnostics


def test_collapsed_representation_has_zero_variance_and_effective_rank():
    Z = np.tile([1.0, 2.0, 3.0], (20, 1))
    variance = diagnostics.embedding_variance(Z)
    assert variance["total"] == pytest.approx(0.0)
    assert diagnostics.effective_rank(Z) == pytest.approx(0.0)
    assert diagnostics.participation_ratio(Z) == pytest.approx(0.0)


def test_isotropic_representation_has_effective_rank_near_dimensionality():
    rng = np.random.default_rng(0)
    D = 5
    Z = rng.normal(size=(5000, D))
    # an isotropic Gaussian's covariance eigenvalues are all approximately
    # equal -> effective rank and participation ratio both approach D.
    assert diagnostics.effective_rank(Z) == pytest.approx(D, rel=0.1)
    assert diagnostics.participation_ratio(Z) == pytest.approx(D, rel=0.1)


def test_rank_one_representation_has_effective_rank_one():
    rng = np.random.default_rng(0)
    direction = np.array([3.0, 4.0])  # unnormalized on purpose
    Z = rng.normal(size=(500, 1)) * direction  # every point on one line
    assert diagnostics.effective_rank(Z) == pytest.approx(1.0, abs=1e-6)


def test_pairwise_distance_distribution_collapses_to_zero_when_collapsed():
    Z = np.tile([1.0, 1.0], (10, 1))
    summary = diagnostics.pairwise_distance_summary(Z)
    assert summary["mean"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(0.0)


def test_terminal_state_separability_high_for_well_separated_clusters():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0], scale=0.1, size=(30, 2))
    cluster_b = rng.normal(loc=[10, 10], scale=0.1, size=(30, 2))
    Z = np.vstack([cluster_a, cluster_b])
    labels = np.array(["a"] * 30 + ["b"] * 30)
    score = diagnostics.terminal_state_separability(Z, labels)
    assert score == pytest.approx(1.0, abs=0.05)


def test_terminal_state_separability_low_for_interleaved_clusters():
    rng = np.random.default_rng(0)
    Z = rng.normal(loc=[0, 0], scale=1.0, size=(60, 2))
    labels = np.array(["a", "b"] * 30)  # random labels, no real structure
    score = diagnostics.terminal_state_separability(Z, labels)
    assert score < 0.2


def test_terminal_state_separability_nan_when_underdetermined():
    Z = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.isnan(diagnostics.terminal_state_separability(Z, ["a", "a"]))
    assert np.isnan(diagnostics.terminal_state_separability(Z, ["a", "b"]))


def test_trustworthiness_and_continuity_are_one_for_identical_spaces():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20, 3))
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    assert diagnostics.trustworthiness(d, d, k=3) == pytest.approx(1.0)
    assert diagnostics.continuity(d, d, k=3) == pytest.approx(1.0)


def test_trustworthiness_and_continuity_hand_worked_example():
    # 4 points on a line (true space) vs. the same 4 points with positions
    # 0 and 3 swapped (embedded space) - one k=1 "intruder" per point.
    # Worked by hand against the standard Venna & Kaski formula:
    # T(k) = 1 - (2 / (n*k*(2n-3k-1))) * sum_i sum_{j intruder} (rank_true(i,j) - k)
    p_true = np.array([0.0, 1.0, 2.2, 3.3])
    p_embedded = np.array([3.3, 1.0, 2.2, 0.0])  # points 0 and 3 swapped

    d_true = np.abs(p_true[:, None] - p_true[None, :])
    d_embedded = np.abs(p_embedded[:, None] - p_embedded[None, :])

    assert diagnostics.trustworthiness(d_true, d_embedded, k=1) == pytest.approx(0.25)
    assert diagnostics.continuity(d_true, d_embedded, k=1) == pytest.approx(0.25)


def test_trustworthiness_nan_when_too_few_points_for_k():
    d = np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]).astype(float)
    assert np.isnan(diagnostics.trustworthiness(d, d, k=2))
