"""Validates geometry.branching (spec Sec.8.7) against the branching/
reconverging synthetic trajectories (spec Sec.22): separation should be near
0 over a shared prefix, rise after a split, and - for the reconverging case
only - fall back down once the branches merge again."""
from __future__ import annotations

import numpy as np
import pytest

from geometry import branching, synthetic


def test_normalized_progress_handles_single_event_case():
    assert branching.normalized_progress(0, 1) == pytest.approx(0.0)
    assert branching.normalized_progress(2, 5) == pytest.approx(0.5)


def test_group_separation_nan_for_fewer_than_two_groups():
    assert np.isnan(branching.group_separation([np.array([[0.0, 0.0]])]))


def test_group_separation_nan_for_degenerate_within_group_scatter():
    # two single-point "groups" -> within-group scatter is exactly 0
    a = np.array([[0.0, 0.0]])
    b = np.array([[5.0, 5.0]])
    assert np.isnan(branching.group_separation([a, b]))


def test_group_separation_increases_with_between_group_distance():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0, 0], scale=0.5, size=(20, 2))
    b_near = rng.normal(loc=[1, 0], scale=0.5, size=(20, 2))
    b_far = rng.normal(loc=[10, 0], scale=0.5, size=(20, 2))
    sep_near = branching.group_separation([a, b_near])
    sep_far = branching.group_separation([a, b_far])
    assert sep_far > sep_near


def test_branching_synthetic_common_prefix_is_indistinguishable():
    data = synthetic.branching(n_traces_per_branch=10, n_common=3, n_branch=4, seed=0)
    a, b = data["branch_a"], data["branch_b"]
    # within the shared common prefix (t=0), the two "branches" are just
    # jittered copies of the same point - separation should be small.
    sep_common = branching.group_separation([a[:, 0, :], b[:, 0, :]])
    assert sep_common < 1.0


def test_branching_synthetic_separation_rises_after_the_split():
    data = synthetic.branching(n_traces_per_branch=10, n_common=3, n_branch=4, separation=3.0, seed=0)
    a, b = data["branch_a"], data["branch_b"]
    sep_common = branching.group_separation([a[:, 0, :], b[:, 0, :]])
    sep_last = branching.group_separation([a[:, -1, :], b[:, -1, :]])
    assert sep_last > sep_common


def test_reconverging_synthetic_separation_rises_then_falls():
    data = synthetic.reconverging(
        n_traces_per_branch=10, n_common=3, n_diverge=3, n_reconverge=3, separation=3.0, seed=0
    )
    a, b = data["branch_a"], data["branch_b"]
    T = a.shape[1]
    peak_t = 3 + 3 - 1  # last index of the diverge phase
    sep_start = branching.group_separation([a[:, 0, :], b[:, 0, :]])
    sep_peak = branching.group_separation([a[:, peak_t, :], b[:, peak_t, :]])
    sep_end = branching.group_separation([a[:, T - 1, :], b[:, T - 1, :]])

    assert sep_peak > sep_start
    assert sep_peak > sep_end  # separation falls back down after reconverging


def test_branch_separation_curve_reflects_branch_then_reconverge_shape():
    data = synthetic.reconverging(
        n_traces_per_branch=10, n_common=3, n_diverge=3, n_reconverge=3, separation=3.0, seed=0
    )
    a, b = data["branch_a"], data["branch_b"]
    n_traces, T, _ = a.shape

    vectors = np.concatenate([a.reshape(-1, 2), b.reshape(-1, 2)], axis=0)
    t_index = np.tile(np.arange(T), n_traces)
    t_index = np.concatenate([t_index, t_index])
    group_labels = np.array(["a"] * (n_traces * T) + ["b"] * (n_traces * T))
    s = branching.normalized_progress(t_index, np.full_like(t_index, T))

    curve = branching.branch_separation_curve(s, group_labels, vectors, n_bins=T)
    assert len(curve) > 0
    values = [curve[k] for k in sorted(curve)]
    # separation must not be monotonically increasing throughout - it has to
    # come back down somewhere, since the branches reconverge.
    assert max(values) > values[0]
    assert max(values) > values[-1]
