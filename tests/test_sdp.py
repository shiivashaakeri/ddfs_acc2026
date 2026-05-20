"""End-to-end tests for the funnel synthesis SDP (Eq. 22)."""

from __future__ import annotations

import numpy as np
import pytest

from data_driven.uncertainty import SegmentUncertainty
from synthesis.sdp import synthesize_funnel


def _make_easy_bundle(seed: int = 0):
    """Construct clean data from a stable linear system with no noise/variation."""
    rng = np.random.default_rng(seed)
    n, m, L = 2, 1, 8
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    B = np.array([[0.0], [1.0]])
    H = rng.standard_normal((n, L))
    Xi = rng.standard_normal((m, L))
    H_plus = A @ H + B @ Xi  # no noise: W = 0
    return SegmentUncertainty(
        H=H, H_plus=H_plus, Xi=Xi,
        beta=1e-6,           # tiny since W = 0
        C=1e-4,              # tiny Jacobian variation
        T_tilde=1.0,
    ), A, B


def test_sdp_solves_easy_problem():
    bundle, A, B = _make_easy_bundle()
    n = bundle.n
    P_min = 0.001 * np.eye(n)  # very loose
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_min, R_max, alpha=0.95)

    # P ≻ 0
    assert np.linalg.eigvalsh(design.P).min() > 0
    # log det P finite and positive (i.e., det P > 1; small ellipsoid)
    assert np.isfinite(design.log_det_P)

    # Closed-loop with the recovered K decays Lyapunov:  α P − A_cl P A_cl^T ⪰ 0
    A_cl = A + B @ design.K
    margin = 0.95 * design.P - A_cl @ design.P @ A_cl.T
    eigs = np.linalg.eigvalsh(0.5 * (margin + margin.T))
    assert eigs.min() > -1e-6, f"Lyapunov decay violated, min eig {eigs.min()}"


def test_sdp_respects_P_min_lower_bound():
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    P_min = 0.5 * np.eye(n)  # restrictive
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_min, R_max, alpha=0.95)
    # P ⪰ P_min
    eigs = np.linalg.eigvalsh(design.P - P_min)
    assert eigs.min() > -1e-6


def test_sdp_respects_R_max_input_feasibility():
    bundle, _, _ = _make_easy_bundle()
    n, m = bundle.n, bundle.m
    P_min = 0.001 * np.eye(n)
    R_max = 0.5 * np.eye(m)  # restrictive
    design = synthesize_funnel(bundle, P_min, R_max, alpha=0.95)
    # The Schur-complement constraint R_max ⪰ L P^{-1} L^T.
    L = design.L
    P = design.P
    margin = R_max - L @ np.linalg.solve(P, L.T)
    assert np.linalg.eigvalsh(0.5 * (margin + margin.T)).min() > -1e-6


def test_sdp_cross_segment_growth_constraint():
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    P_prev = 0.5 * np.eye(n)
    mu = 1.05
    P_min = 0.001 * np.eye(n)
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_min, R_max, alpha=0.95, P_prev=P_prev, mu=mu)
    margin = mu * P_prev - design.P
    assert np.linalg.eigvalsh(0.5 * (margin + margin.T)).min() > -1e-6


def test_sdp_recovers_K_from_L():
    """K = L P^{-1} should be returned correctly, with L = K P self-consistent."""
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    design = synthesize_funnel(bundle, 0.001 * np.eye(n), 100.0 * np.eye(bundle.m), alpha=0.95)
    assert np.allclose(design.L, design.K @ design.P, atol=1e-6)
