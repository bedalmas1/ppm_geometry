"""Per-trace trajectory geometry metrics (spec Sec.8.1-8.5).

Every function takes one trace's latent trajectory Z, shape (T, D) - z_1..z_T
in order - and returns either a scalar (straightness, path_length) or a
per-transition array (velocity, curvature, smoothness). Degenerate cases (too
few points, near-zero-norm displacement) return NaN rather than raising, so a
caller aggregating over many traces can nanmean/nanmedian and see how many
traces were affected, instead of a single bad trace crashing a whole run.

Straightness alone must never be reported without the collapse/triviality
diagnostics in geometry.diagnostics (spec Sec.9) - a collapsed trace
(z_1~=...~=z_T) makes straightness NaN here (0/0), not artificially 1, but a
*nearly*-collapsed trace can still read as spuriously straight, which only
the variance/effective-rank diagnostics can catch.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _as_trajectory(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"expected a (T, D) trajectory array, got shape {z.shape}")
    return z


def _displacements(z: np.ndarray) -> np.ndarray:
    """Delta_t = z_{t+1} - z_t, for t=1..T-1. Shape (T-1, D)."""
    return np.diff(z, axis=0)


def path_length(z) -> float:
    """L(sigma) = sum_{t=1}^{T-1} ||z_{t+1} - z_t||_2 (spec Sec.8.2). 0.0 for
    a trace with fewer than 2 points (no transitions to sum)."""
    z = _as_trajectory(z)
    if len(z) < 2:
        return 0.0
    return float(np.linalg.norm(_displacements(z), axis=1).sum())


def straightness(z) -> float:
    """S(sigma) = ||z_T - z_1||_2 / L(sigma) (spec Sec.8.1).

    NaN if T<2 (no path at all) or L(sigma)==0 (collapsed representation,
    z_1=...=z_T - a genuine 0/0, must never be silently read as 1)."""
    z = _as_trajectory(z)
    if len(z) < 2:
        return float("nan")
    length = path_length(z)
    if length <= _EPS:
        return float("nan")
    net_displacement = np.linalg.norm(z[-1] - z[0])
    return float(net_displacement / length)


def velocity(z) -> np.ndarray:
    """v_t = ||z_t - z_{t-1}|| (spec Sec.8.3), for t=2..T. Shape (T-1,);
    velocity(z)[0] is v_2. Empty array for a trace with fewer than 2 points."""
    z = _as_trajectory(z)
    if len(z) < 2:
        return np.array([])
    return np.linalg.norm(_displacements(z), axis=1)


def curvature(z) -> np.ndarray:
    """theta_t = arccos(Delta_t . Delta_{t+1} / (||Delta_t|| ||Delta_{t+1}||))
    (spec Sec.8.4), for t=2..T-1. Shape (T-2,); curvature(z)[0] is theta_2.

    NaN wherever either adjacent displacement has (near-)zero norm: the
    direction at a stationary point is undefined, so this is deliberately
    NOT treated as zero curvature ("no turn"). Cosine argument is clipped to
    [-1, 1] before arccos to absorb floating-point overshoot."""
    z = _as_trajectory(z)
    if len(z) < 3:
        return np.array([])
    deltas = _displacements(z)
    d1, d2 = deltas[:-1], deltas[1:]
    n1 = np.linalg.norm(d1, axis=1)
    n2 = np.linalg.norm(d2, axis=1)
    denom = n1 * n2
    dot = np.sum(d1 * d2, axis=1)

    theta = np.full(len(denom), np.nan)
    valid = denom > _EPS
    cos_theta = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    theta[valid] = np.arccos(cos_theta)
    return theta


def smoothness(z) -> np.ndarray:
    """a_t = ||(z_{t+1}-z_t) - (z_t-z_{t-1})|| (spec Sec.8.5), for t=2..T-1.
    Shape (T-2,); smoothness(z)[0] is a_2. This is a discrete second
    difference (acceleration proxy), well-defined even where curvature is
    NaN (it doesn't involve a division by displacement norms)."""
    z = _as_trajectory(z)
    if len(z) < 3:
        return np.array([])
    deltas = _displacements(z)
    accel = deltas[1:] - deltas[:-1]
    return np.linalg.norm(accel, axis=1)
