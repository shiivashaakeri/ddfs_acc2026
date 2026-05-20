"""Tests for data_driven: excitation, β, persistence, uncertainty bundle."""

from __future__ import annotations

import numpy as np

from data_driven.beta import beta_segment
from data_driven.data_collection import (
    BoundedExcitation,
    SegmentData,
    assemble_data_matrices,
    persistently_exciting,
)
from data_driven.uncertainty import SegmentUncertainty


def test_excitation_bound():
    exc = BoundedExcitation(eps_bar=0.15, dim=2, rng=np.random.default_rng(0))
    samples = exc.sequence(500)
    norms = np.linalg.norm(samples, axis=1)
    assert norms.max() <= 0.15 + 1e-9
    # Sanity: distribution not collapsed to a point.
    assert norms.min() < 0.14


def _toy_segment(L: int, n: int = 4, m: int = 2, seed: int = 0, noise: float = 0.0):
    """Generate a synthetic segment with known (A, B) and bounded noise."""
    rng = np.random.default_rng(seed)
    A = np.eye(n) + 0.1 * rng.standard_normal((n, n))
    B = 0.1 * rng.standard_normal((n, m))
    eta = rng.standard_normal((L, n))
    xi = rng.standard_normal((L, m))
    w = noise * rng.standard_normal((L, n))
    eta_next = (eta @ A.T) + (xi @ B.T) + w
    seg = SegmentData(eta=eta, xi=xi, eta_next=eta_next, k_start=0, k_segment_start=0)
    return seg, A, B, w


def test_data_equation_holds_exactly_when_noise_zero():
    seg, A, B, _ = _toy_segment(L=20, noise=0.0)
    H, H_plus, Xi = assemble_data_matrices(seg)
    residual = H_plus - A @ H - B @ Xi
    assert np.allclose(residual, 0.0, atol=1e-10)


def test_persistence_of_excitation():
    seg, _, _, _ = _toy_segment(L=20)
    H, _, Xi = assemble_data_matrices(seg)
    assert persistently_exciting(H, Xi)


def test_persistence_fails_when_data_low_rank():
    n, m, L = 4, 2, 10
    H = np.tile(np.array([[1.0], [1.0], [1.0], [1.0]]), (1, L))  # rank-1
    Xi = np.zeros((m, L))
    assert not persistently_exciting(H, Xi)


def test_beta_bounds_W_energy():
    # Construct W bounded by the Lemma 3 pointwise formula, then verify β ≥ ||W||_F².
    rng = np.random.default_rng(7)
    L, n, m = 30, 4, 2
    eta = 0.3 * rng.standard_normal((L, n))
    xi = 0.2 * rng.standard_normal((L, m))
    seg = SegmentData(eta=eta, xi=xi, eta_next=eta, k_start=5, k_segment_start=0)

    C, gamma, L_r = 3.64, 0.034, 0.074
    beta = beta_segment(seg, C=C, gamma=gamma, L_r=L_r)

    # Synthesize a W whose pointwise norms saturate the Lemma 3 bound.
    z_norm = np.linalg.norm(np.hstack([eta, xi]), axis=1)
    dt = seg.time_offsets()
    bounds = C * np.abs(dt) * z_norm + gamma + L_r * z_norm ** 2
    W = bounds[None, :] * (rng.standard_normal((n, L)))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    W *= bounds[None, :]

    # ||W||_F² = sum of ||w_k||² where each ||w_k|| ≤ bound_k.
    energy = float(np.sum(np.linalg.norm(W, axis=0) ** 2))
    assert energy <= beta + 1e-6, f"energy {energy} should be ≤ β = {beta}"
    # And W W^T should have spectral norm ≤ β (since W W^T ⪯ ||W||_F² I).
    lam_max = float(np.linalg.eigvalsh(W @ W.T).max())
    assert lam_max <= beta + 1e-6


def test_uncertainty_bundle_shapes():
    seg, _, _, _ = _toy_segment(L=20)
    H, H_plus, Xi = assemble_data_matrices(seg)
    bundle = SegmentUncertainty(H=H, H_plus=H_plus, Xi=Xi, beta=1.0, C=3.64, T_tilde=199.0)
    assert bundle.n == 4
    assert bundle.m == 2
    assert bundle.L == 20
