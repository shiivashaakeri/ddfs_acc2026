"""Tests for finite-difference Jacobians and Lipschitz estimators."""

from __future__ import annotations

import numpy as np

from dynamics.digital_twin import create_plant
from linearization.jacobians import jacobians_along_trajectory, jacobians_discrete
from linearization.lipschitz import estimate_gamma, estimate_L_J, estimate_L_r


def test_jacobian_recovers_identity_for_identity_map():
    def id_step(x, u):
        return np.asarray(x, dtype=float)

    A, B = jacobians_discrete(id_step, np.zeros(4), np.zeros(2))
    assert np.allclose(A, np.eye(4), atol=1e-6)
    assert np.allclose(B, np.zeros((4, 2)), atol=1e-6)


def test_jacobian_recovers_linear_map():
    A_true = np.array([[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.1], [0.0, 0.0, 0.9, 0.0], [0.0, 0.0, 0.0, 0.9]])
    B_true = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])

    def lin_step(x, u):
        return A_true @ x + B_true @ u

    A, B = jacobians_discrete(lin_step, np.zeros(4), np.zeros(2))
    assert np.allclose(A, A_true, atol=1e-6)
    assert np.allclose(B, B_true, atol=1e-6)


def test_first_order_taylor_residual_small():
    plant = create_plant()
    rng = np.random.default_rng(42)
    x0 = np.array([0.3, -0.2, 0.0, 0.0])
    u0 = np.array([0.5, -0.5])
    A, B = jacobians_discrete(plant.step, x0, u0)
    f0 = plant.step(x0, u0)
    for _ in range(20):
        dx = rng.normal(size=4) * 1e-4
        du = rng.normal(size=2) * 1e-4
        f1 = plant.step(x0 + dx, u0 + du)
        lin = f0 + A @ dx + B @ du
        # Residual is O(||z||^2); for ||z|| ~ 1e-4 expect ~1e-8.
        assert np.linalg.norm(f1 - lin) < 1e-6


def test_along_trajectory_shapes():
    plant = create_plant()
    x_nom = np.tile(np.array([0.1, -0.1, 0.0, 0.0]), (5, 1))
    u_nom = np.zeros((4, 2))
    A_seq, B_seq = jacobians_along_trajectory(plant.step, x_nom, u_nom)
    assert A_seq.shape == (4, 4, 4)
    assert B_seq.shape == (4, 4, 2)


def test_lipschitz_estimators_finite_and_positive():
    from dynamics.digital_twin import create_twin

    plant = create_plant()
    twin = create_twin()
    # Tiny nominal stub: 10 steps around a working point
    x_nom = np.tile(np.array([0.3, -0.2, 0.0, 0.0]), (11, 1))
    u_nom = np.tile(np.array([-8.5, -2.4]), (10, 1))

    gamma = estimate_gamma(plant.step, twin.step, x_nom, u_nom)
    L_J = estimate_L_J(plant.step, x_nom, u_nom, radius=0.05, n_probes=4)
    L_r = estimate_L_r(plant.step, x_nom, u_nom, radius=0.05, n_probes=4)

    assert 0 < gamma < 1.0
    assert 0 < L_J < 1000.0
    assert 0 < L_r < 100.0
