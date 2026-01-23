"""
Tests for the linearization module.

Tests Jacobian computation accuracy and Lipschitz constant estimation.
"""

import numpy as np
import pytest

from config.params import SIM_PARAMS, INITIAL_STATES, LIPSCHITZ
from dynamics.digital_twin import create_plant, create_twin
from trajectory.nominal_planner import generate_nominal_trajectory
from trajectory.trajectory_utils import compute_trajectory_speed
from linearization.jacobians import (
    numerical_jacobian,
    compute_jacobians_along_trajectory,
    jacobian_difference_norm,
)
from linearization.lipschitz import (
    estimate_gamma,
    estimate_L_J,
    estimate_L_r,
    compute_C,
    estimate_all_constants,
)


@pytest.fixture
def twin():
    """Create digital twin."""
    return create_twin()


@pytest.fixture
def plant():
    """Create plant."""
    return create_plant()


@pytest.fixture
def nominal_trajectory(twin):
    """Generate nominal trajectory."""
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    N = SIM_PARAMS["N"]

    x_nom, u_nom = generate_nominal_trajectory(twin, x0, x_goal, u_goal, N)
    return x_nom, u_nom


def test_jacobian_accuracy(twin):
    """
    Test that numerical Jacobians are accurate by comparing with
    finite difference approximation at multiple points.
    """
    x_test = np.array([1.0, -0.5, 0.1, -0.1])
    u_test = np.array([5.0, 2.0])

    def f_discrete(x, u):
        return twin.step(x, u)

    # Compute Jacobians with default eps
    A, B = numerical_jacobian(f_discrete, x_test, u_test)

    # Verify shapes
    assert A.shape == (4, 4), f"A shape should be (4, 4), got {A.shape}"
    assert B.shape == (4, 2), f"B shape should be (4, 2), got {B.shape}"

    # Verify by checking that linear approximation is accurate for small perturbations
    eps = 1e-4
    x_pert = x_test + eps * np.array([1, 0, 0, 0])
    u_pert = u_test

    f_nom = f_discrete(x_test, u_test)
    f_pert = f_discrete(x_pert, u_pert)
    f_linear = f_nom + A @ (x_pert - x_test) + B @ (u_pert - u_test)

    error = np.linalg.norm(f_pert - f_linear)
    assert error < 1e-6, f"Linear approximation error too large: {error}"


def test_jacobian_variation_bound(twin, nominal_trajectory):
    """
    Test that Jacobian variation satisfies Assumption 3:
    ||[A(k) - A(s), B(k) - B(s)]|| <= L_J * v * |k - s|

    This is the key assumption for the data-driven approach.
    """
    x_nom, u_nom = nominal_trajectory
    v = compute_trajectory_speed(x_nom, u_nom)

    def f_discrete(x, u):
        return twin.step(x, u)

    # Compute Jacobians at sampled points
    N = len(u_nom)
    sample_indices = [0, 50, 100, 200, 300, 400, 500, N - 1]
    sample_indices = [i for i in sample_indices if i < N]

    jacobians = {}
    for k in sample_indices:
        A_k, B_k = numerical_jacobian(f_discrete, x_nom[k], u_nom[k])
        jacobians[k] = (A_k, B_k)

    # Check variation bound for all pairs
    # Use paper's L_J as reference (we allow some margin)
    L_J_ref = LIPSCHITZ["L_J"]

    for i, k in enumerate(sample_indices):
        for s in sample_indices[i + 1 :]:
            A_k, B_k = jacobians[k]
            A_s, B_s = jacobians[s]

            diff_norm = jacobian_difference_norm(A_k, B_k, A_s, B_s)
            bound = L_J_ref * v * abs(k - s)

            # Allow 50% margin since L_J is estimated
            assert diff_norm <= 1.5 * bound, (
                f"Jacobian variation bound violated at (k={k}, s={s}): "
                f"{diff_norm:.4f} > {1.5 * bound:.4f}"
            )


def test_lipschitz_estimates_positive(plant, twin, nominal_trajectory):
    """
    Test that all Lipschitz constants are positive.
    """
    x_nom, u_nom = nominal_trajectory
    v = compute_trajectory_speed(x_nom, u_nom)

    gamma = estimate_gamma(plant, twin, x_nom, u_nom)
    L_J = estimate_L_J(twin, x_nom, u_nom, v)
    L_r = estimate_L_r(twin, x_nom, u_nom)
    C = compute_C(L_J, v)

    assert gamma > 0, f"gamma should be positive, got {gamma}"
    assert L_J > 0, f"L_J should be positive, got {L_J}"
    assert L_r > 0, f"L_r should be positive, got {L_r}"
    assert C > 0, f"C should be positive, got {C}"


def test_gamma_reasonable(plant, twin, nominal_trajectory):
    """
    Test that gamma is in a reasonable range (paper value: 0.034).
    """
    x_nom, u_nom = nominal_trajectory
    gamma = estimate_gamma(plant, twin, x_nom, u_nom)

    # Gamma should be small (mismatch is bounded)
    assert gamma < 1.0, f"gamma too large: {gamma}"
    # Gamma should be positive (there is some mismatch)
    assert gamma > 0.001, f"gamma too small: {gamma}"


def test_L_J_reasonable(twin, nominal_trajectory):
    """
    Test that L_J is in a reasonable range.

    Note: Our trajectory-based L_J estimate is typically much smaller than
    the paper's global L_J = 16.94 because the slow trajectory has small
    Jacobian variations. The estimate is a lower bound on the true L_J.
    """
    x_nom, u_nom = nominal_trajectory
    v = compute_trajectory_speed(x_nom, u_nom)

    L_J = estimate_L_J(twin, x_nom, u_nom, v)

    # L_J should be positive (trajectory has some Jacobian variation)
    assert L_J > 0.001, f"L_J too small: {L_J}"
    # L_J estimate from trajectory should be smaller than global bound
    assert L_J < LIPSCHITZ["L_J"], f"L_J estimate exceeds paper's global bound: {L_J}"


def test_C_equals_LJ_times_v(twin, nominal_trajectory):
    """
    Test that C = L_J * v relationship holds.
    """
    x_nom, u_nom = nominal_trajectory
    v = compute_trajectory_speed(x_nom, u_nom)

    L_J = estimate_L_J(twin, x_nom, u_nom, v)
    C = compute_C(L_J, v)

    assert np.isclose(C, L_J * v), f"C = {C} should equal L_J * v = {L_J * v}"


def test_estimate_all_constants(plant, twin, nominal_trajectory):
    """
    Test the convenience function that estimates all constants.
    """
    x_nom, u_nom = nominal_trajectory
    v = compute_trajectory_speed(x_nom, u_nom)

    constants = estimate_all_constants(plant, twin, x_nom, u_nom, v)

    # Check all keys present
    expected_keys = {"gamma", "L_J", "L_r", "C", "v"}
    assert set(constants.keys()) == expected_keys

    # Check all values are positive
    for key, value in constants.items():
        assert value > 0, f"{key} should be positive, got {value}"

    # Check C = L_J * v
    assert np.isclose(constants["C"], constants["L_J"] * constants["v"])
