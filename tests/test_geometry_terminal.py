"""Validates geometry.terminal (spec Sec.8.6) - progress toward terminal
regions, via both the Euclidean-centroid prototype and the k-NN
neighborhood-based alternative the spec requires evaluating alongside it."""
from __future__ import annotations

import numpy as np
import pytest

from geometry import terminal


def test_terminal_centroids_match_hand_computed_means():
    z_terminal = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
    labels = np.array(["y", "y", "n", "n"])
    centroids = terminal.terminal_centroids(z_terminal, labels)
    assert np.allclose(centroids["y"], [1.0, 0.0])
    assert np.allclose(centroids["n"], [11.0, 10.0])


def test_distance_to_prototype_euclidean():
    z = np.array([[0.0, 0.0], [3.0, 4.0]])  # distance 5 from origin
    prototype = np.array([0.0, 0.0])
    assert np.allclose(terminal.distance_to_prototype(z, prototype), [0.0, 5.0])


def test_approach_trajectory_has_decreasing_distance_and_negative_trend():
    # a trace that walks straight toward its own terminal point
    z = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [10.0, 0.0]])
    prototype = z[-1]
    distances = terminal.distance_to_prototype(z, prototype)
    assert np.all(np.diff(distances) < 0)  # strictly decreasing
    assert terminal.progress_trend(distances) == pytest.approx(-1.0)


def test_progress_trend_nan_for_constant_distance():
    assert np.isnan(terminal.progress_trend([1.0, 1.0, 1.0]))


def test_knn_distance_to_class_matches_manual_computation_for_k_equal_class_size():
    z_terminal = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
    labels = np.array(["y", "y", "n"])
    z_query = np.array([[1.0, 0.0]])  # equidistant (1.0) from both "y" points
    dist = terminal.knn_distance_to_class(z_query, z_terminal, labels, "y", k=2)
    assert dist[0] == pytest.approx(1.0)


def test_knn_distance_to_class_survives_multimodal_terminal_region():
    # two well-separated clusters sharing one label - a Euclidean centroid
    # would land at their meaningless midpoint; the k-NN alternative
    # correctly reports "close to the nearest real cluster" instead.
    cluster_1 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    cluster_2 = np.array([[100.0, 0.0], [100.0, 1.0], [101.0, 0.0]])
    z_terminal = np.vstack([cluster_1, cluster_2])
    labels = np.array(["y"] * 3 + ["y"] * 3)
    query_near_cluster_1 = np.array([[0.5, 0.5]])

    centroid = terminal.terminal_centroids(z_terminal, labels)["y"]
    centroid_distance = np.linalg.norm(query_near_cluster_1[0] - centroid)
    knn_distance = terminal.knn_distance_to_class(
        query_near_cluster_1, z_terminal, labels, "y", k=3
    )[0]

    assert knn_distance < 2.0  # genuinely close to cluster 1
    assert centroid_distance > 40.0  # the single centroid is nowhere near either cluster
