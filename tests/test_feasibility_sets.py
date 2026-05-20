"""Tests for MVIE feasibility ellipsoids."""

from __future__ import annotations

import numpy as np

from config.params import INITIAL_STATES, INPUT_BOUNDS, STATE_BOUNDS
from synthesis.feasibility_sets import Pmin_from_box, mvie_from_box


def _state_box():
    lo = np.array([STATE_BOUNDS[k][0] for k in ["q1", "q2", "dq1", "dq2"]])
    hi = np.array([STATE_BOUNDS[k][1] for k in ["q1", "q2", "dq1", "dq2"]])
    return lo, hi


def test_mvie_matches_paper_at_k0():
    # Paper Section VI: P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14).
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    lo, hi = _state_box()
    P = Pmin_from_box(x0, lo, hi)
    w, V = np.linalg.eigh(P)
    Psqrt = V @ np.diag(np.sqrt(w)) @ V.T
    diag = np.diag(Psqrt)
    expected = np.array([0.19, 0.12, 0.12, 0.14])
    # Loose tolerance (paper rounds to 2 sig figs).
    assert np.allclose(diag, expected, atol=0.02), f"got {diag}"


def test_mvie_is_inscribed():
    # Every η on the boundary of {η^T P η <= 1} must lie inside the shifted box.
    rng = np.random.default_rng(0)
    center = np.array([0.28, -0.22, 0.0, 0.0])
    lo, hi = _state_box()
    P = Pmin_from_box(center, lo, hi)
    # Sample boundary points: η = c P^{-1/2} u with ||u|| = 1
    w, V = np.linalg.eigh(P)
    Pinv_half = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    for _ in range(200):
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        eta = Pinv_half @ u
        x = center + eta
        assert (x >= lo - 1e-6).all() and (x <= hi + 1e-6).all()


def test_mvie_input_box():
    u0 = np.array(INITIAL_STATES["u_goal"])
    u_lo = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])
    Z = mvie_from_box(u0, u_lo, u_hi)
    # Z should be diagonal (axis-aligned box) with entries == min face-slacks
    expected = np.diag([min(u_hi[0] - u0[0], u0[0] - u_lo[0]),
                        min(u_hi[1] - u0[1], u0[1] - u_lo[1])])
    assert np.allclose(Z, expected, atol=0.05)
