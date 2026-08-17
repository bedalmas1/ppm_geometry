"""Validates geometry.trajectory (spec Sec.8.1-8.5) against the synthetic
suite (spec Sec.22) - every case has an analytically-known expected value,
computed independently in this file rather than hardcoded magic numbers."""
from __future__ import annotations

import numpy as np
import pytest

from geometry import synthetic, trajectory


def test_straight_line_has_straightness_exactly_one():
    z = synthetic.straight_line(n_points=5)
    assert trajectory.straightness(z) == pytest.approx(1.0)


def test_straight_line_has_zero_curvature_everywhere():
    z = synthetic.straight_line(n_points=6)
    theta = trajectory.curvature(z)
    assert len(theta) == 4
    assert np.allclose(theta, 0.0, atol=1e-8)


def test_straight_line_has_zero_smoothness_everywhere():
    z = synthetic.straight_line(n_points=6)
    a = trajectory.smoothness(z)
    assert np.allclose(a, 0.0, atol=1e-8)


def test_straight_line_path_length_and_velocity():
    z = synthetic.straight_line(n_points=5, step=2.0)
    assert trajectory.path_length(z) == pytest.approx(8.0)  # 4 steps of size 2
    assert np.allclose(trajectory.velocity(z), 2.0)


def test_circular_arc_straightness_matches_chord_over_chord_sum():
    n_points, radius, total_angle = 9, 1.0, np.pi
    z = synthetic.circular_arc(n_points=n_points, radius=radius, total_angle=total_angle)

    dtheta = total_angle / (n_points - 1)
    expected_chord = 2 * radius * np.sin(total_angle / 2)
    expected_path = (n_points - 1) * 2 * radius * np.sin(dtheta / 2)
    expected_straightness = expected_chord / expected_path

    assert trajectory.path_length(z) == pytest.approx(expected_path)
    assert trajectory.straightness(z) == pytest.approx(expected_straightness)
    assert trajectory.straightness(z) < 1.0


def test_circular_arc_has_constant_curvature_equal_to_angular_step():
    n_points, total_angle = 9, np.pi
    z = synthetic.circular_arc(n_points=n_points, total_angle=total_angle)
    dtheta = total_angle / (n_points - 1)

    theta = trajectory.curvature(z)
    assert len(theta) == n_points - 2
    assert np.allclose(theta, dtheta, atol=1e-8)


def test_backtracking_has_one_exact_reversal_and_known_straightness():
    z = synthetic.backtracking(n_forward=3, n_backward=3, step=1.0)
    # forward: 0,1,2 ; backward: 1,0,-1 -> path length 2+3=5, net |−1−0|=1
    assert trajectory.path_length(z) == pytest.approx(5.0)
    assert trajectory.straightness(z) == pytest.approx(0.2)

    theta = trajectory.curvature(z)
    assert len(theta) == 4
    # exactly one full reversal (pi), everywhere else perfectly straight (0)
    assert np.sum(np.isclose(theta, np.pi, atol=1e-8)) == 1
    assert np.sum(np.isclose(theta, 0.0, atol=1e-8)) == 3


def test_collapsed_representation_has_zero_path_length_and_nan_straightness():
    z = synthetic.collapsed(n_points=5, dim=3)
    assert trajectory.path_length(z) == pytest.approx(0.0)
    assert np.isnan(trajectory.straightness(z))
    assert np.allclose(trajectory.velocity(z), 0.0)


def test_collapsed_representation_curvature_is_undefined_not_zero():
    # A stationary point's direction is undefined - must be NaN, never
    # silently read as "no turn" (curvature 0).
    z = synthetic.collapsed(n_points=5, dim=2)
    theta = trajectory.curvature(z)
    assert np.all(np.isnan(theta))


def test_too_short_trajectories_return_empty_or_nan_not_raise():
    single_point = np.array([[1.0, 2.0]])
    assert trajectory.path_length(single_point) == pytest.approx(0.0)
    assert np.isnan(trajectory.straightness(single_point))
    assert trajectory.velocity(single_point).size == 0
    assert trajectory.curvature(single_point).size == 0
    assert trajectory.smoothness(single_point).size == 0

    two_points = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert trajectory.curvature(two_points).size == 0
    assert trajectory.smoothness(two_points).size == 0


def test_smoothness_detects_a_sharp_kink_but_not_a_gentle_one():
    straight = synthetic.straight_line(n_points=4)
    kinked = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    assert np.allclose(trajectory.smoothness(straight), 0.0)
    assert np.all(trajectory.smoothness(kinked) > 0.0)
