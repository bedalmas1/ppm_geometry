"""Synthetic latent-trajectory suite (spec Sec.22).

Every case here has an analytically-known expected geometry - constructed
and validated against geometry.trajectory/geometry.branching *before* any
metric ever touches a real cached embedding, per PLAN.md's Phase 5
pilot-first rule ("Do this before analyzing any real model").
"""
from __future__ import annotations

import numpy as np


def straight_line(n_points: int = 5, step: float = 1.0) -> np.ndarray:
    """z_t = (t, 0) (spec's literal example): straightness must be exactly
    1.0, curvature exactly 0 at every interior point (no direction change at
    all along a straight line)."""
    t = np.arange(n_points, dtype=float) * step
    return np.column_stack([t, np.zeros_like(t)])


def circular_arc(n_points: int = 9, radius: float = 1.0, total_angle: float = np.pi) -> np.ndarray:
    """Points at equal angular steps along a circular arc: straightness < 1
    (a chord is always shorter than the arc it subtends), and - because
    consecutive chords of an equally-spaced inscribed polygon all subtend
    the same angle - curvature is exactly constant and equal to the angular
    step dtheta = total_angle / (n_points - 1) at every interior point."""
    angles = np.linspace(0.0, total_angle, n_points)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def backtracking(n_forward: int = 3, n_backward: int = 3, step: float = 1.0) -> np.ndarray:
    """Moves toward a target for `n_forward` points, then reverses back over
    the same line for `n_backward` points (spec's "backtracking" case):
    exactly one interior point has a full reversal (curvature exactly pi,
    the sharpest possible direction change), every other interior point has
    curvature exactly 0 (continuing straight in the same direction)."""
    forward = np.arange(n_forward, dtype=float) * step
    peak = forward[-1]
    backward = peak - np.arange(1, n_backward + 1, dtype=float) * step
    xs = np.concatenate([forward, backward])
    return np.column_stack([xs, np.zeros_like(xs)])


def collapsed(n_points: int = 5, dim: int = 2, value=None) -> np.ndarray:
    """z_t = c for every t (spec's literal example): path length exactly 0,
    straightness undefined (0/0 -> NaN, never silently read as 1) - exactly
    the pathology geometry.diagnostics' collapse guard exists to catch."""
    c = np.zeros(dim) if value is None else np.asarray(value, dtype=float)
    return np.tile(c, (n_points, 1))


def branching(
    n_traces_per_branch: int = 6,
    n_common: int = 3,
    n_branch: int = 4,
    separation: float = 2.0,
    jitter: float = 0.05,
    seed: int = 0,
) -> dict:
    """A common prefix shared by every trace, followed by two diverging
    groups of trajectories (spec's "branching trajectories"). Each side has
    `n_traces_per_branch` parallel cases with small per-trace jitter, so
    within-branch scatter is nonzero and geometry.branching.group_separation
    has something non-degenerate to measure: separation should read ~0 over
    the shared prefix and rise once the split happens.

    Returns {"branch_a": (n_traces_per_branch, T, 2), "branch_b": same}.
    """
    rng = np.random.default_rng(seed)
    t_common = np.arange(n_common, dtype=float)
    t_branch = np.arange(1, n_branch + 1, dtype=float)

    def _make(sign: float) -> np.ndarray:
        traces = []
        for _ in range(n_traces_per_branch):
            common = np.column_stack([t_common, rng.normal(0, jitter, n_common)])
            branch = np.column_stack(
                [
                    n_common - 1 + t_branch,
                    sign * separation * t_branch + rng.normal(0, jitter, n_branch),
                ]
            )
            traces.append(np.vstack([common, branch]))
        return np.stack(traces)

    return {"branch_a": _make(+1.0), "branch_b": _make(-1.0)}


def reconverging(
    n_traces_per_branch: int = 6,
    n_common: int = 3,
    n_diverge: int = 3,
    n_reconverge: int = 3,
    separation: float = 2.0,
    jitter: float = 0.05,
    seed: int = 0,
) -> dict:
    """Two branches that separate and then merge back together (spec's
    "reconverging trajectories"): extends geometry.synthetic.branching with
    a merge phase that linearly returns each trace's y-offset to 0 - group
    separation should rise after the split, then fall back toward 0 by the
    final shared point."""
    branches = branching(n_traces_per_branch, n_common, n_diverge, separation, jitter, seed)
    t_merge = np.arange(1, n_reconverge + 1, dtype=float)
    rng = np.random.default_rng(seed + 1)

    def _merge(traces: np.ndarray) -> np.ndarray:
        merged = []
        for trace in traces:
            last_x, last_y = trace[-1]
            merge_x = last_x + t_merge
            merge_y = last_y * (1 - t_merge / n_reconverge) + rng.normal(0, jitter, n_reconverge)
            merged.append(np.vstack([trace, np.column_stack([merge_x, merge_y])]))
        return np.stack(merged)

    return {"branch_a": _merge(branches["branch_a"]), "branch_b": _merge(branches["branch_b"])}
