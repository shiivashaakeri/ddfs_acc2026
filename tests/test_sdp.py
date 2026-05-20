"""End-to-end tests for the funnel synthesis SDP, shape-matrix convention."""

from __future__ import annotations

import numpy as np

from data_driven.uncertainty import SegmentUncertainty
from synthesis.sdp import synthesize_funnel


def _make_easy_bundle(seed: int = 0):
    """Clean data from a stable linear system with no noise / variation."""
    rng = np.random.default_rng(seed)
    n, m, L = 2, 1, 8
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    B = np.array([[0.0], [1.0]])
    H = rng.standard_normal((n, L))
    Xi = rng.standard_normal((m, L))
    H_plus = A @ H + B @ Xi  # no noise
    return SegmentUncertainty(
        H=H, H_plus=H_plus, Xi=Xi,
        beta=1e-6, C=1e-4, T_tilde=1.0,
    ), A, B


def test_sdp_solves_easy_problem():
    bundle, A, B = _make_easy_bundle()
    n = bundle.n
    P_max_shape = 1000.0 * np.eye(n)  # loose
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_max_shape, R_max, alpha=0.95)

    assert np.linalg.eigvalsh(design.P).min() > 0
    assert np.isfinite(design.log_det_P)

    # Lyapunov decay in shape convention: A_cl P A_cl^T ⪯ α P.
    A_cl = A + B @ design.K
    margin = 0.95 * design.P - A_cl @ design.P @ A_cl.T
    eigs = np.linalg.eigvalsh(0.5 * (margin + margin.T))
    assert eigs.min() > -1e-6, f"Lyapunov decay violated, min eig {eigs.min()}"


def test_sdp_respects_P_max_shape_upper_bound():
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    P_max_shape = 0.5 * np.eye(n)  # restrictive
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_max_shape, R_max, alpha=0.95)
    eigs = np.linalg.eigvalsh(P_max_shape - design.P)
    assert eigs.min() > -1e-6


def test_sdp_respects_R_max_input_feasibility():
    bundle, _, _ = _make_easy_bundle()
    n, m = bundle.n, bundle.m
    P_max_shape = 1000.0 * np.eye(n)
    R_max = 0.5 * np.eye(m)
    design = synthesize_funnel(bundle, P_max_shape, R_max, alpha=0.95)
    L = design.L
    P = design.P
    margin = R_max - L @ np.linalg.solve(P, L.T)
    assert np.linalg.eigvalsh(0.5 * (margin + margin.T)).min() > -1e-6


def test_sdp_cross_segment_growth_constraint():
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    P_prev = 1.0 * np.eye(n)
    mu = 1.05
    P_max_shape = 1000.0 * np.eye(n)
    R_max = 100.0 * np.eye(bundle.m)
    design = synthesize_funnel(bundle, P_max_shape, R_max, alpha=0.95, P_prev=P_prev, mu=mu)
    # Shape convention: P ⪰ (1/μ) P_prev.
    margin = design.P - (1.0 / mu) * P_prev
    assert np.linalg.eigvalsh(0.5 * (margin + margin.T)).min() > -1e-6


def test_sdp_recovers_K_from_L():
    bundle, _, _ = _make_easy_bundle()
    n = bundle.n
    design = synthesize_funnel(bundle, 1000.0 * np.eye(n), 100.0 * np.eye(bundle.m), alpha=0.95)
    assert np.allclose(design.L, design.K @ design.P, atol=1e-6)
