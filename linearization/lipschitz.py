"""
Lipschitz Constant Estimation.

Estimate constants from the paper for robust funnel synthesis:
- gamma: Plant-twin mismatch bound (Eq. 4)
- L_J: Jacobian Lipschitz constant (Assumption 3)
- L_r: Second-order remainder Lipschitz constant
- C = L_J * v: Jacobian variation bound

Paper values (Table I / Section VI):
- gamma = 0.034
- L_J = 16.94
- L_r = 0.074
- C = 3.64
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from linearization.jacobians import (
    compute_jacobians_along_trajectory,
    jacobian_difference_norm,
    numerical_jacobian,
)


def estimate_gamma(
    plant,
    twin,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
) -> float:
    """
    Estimate the plant-twin mismatch bound gamma.

    gamma = max_{k} ||f_plant(x_nom(k), u_nom(k)) - f_twin(x_nom(k), u_nom(k))||

    This is the maximum norm of Delta(x,u) = f_plant - f_twin along
    the nominal trajectory.

    Args:
        plant: Plant DiscreteSystem with step() method
        twin: Twin DiscreteSystem with step() method
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory

    Returns:
        gamma: Estimated mismatch bound (paper value: 0.034)
    """
    N = len(u_nom)
    max_mismatch = 0.0

    for k in range(N):
        x_k = x_nom[k]
        u_k = u_nom[k]

        # Compute one-step predictions
        x_plant_next = plant.step(x_k, u_k)
        x_twin_next = twin.step(x_k, u_k)

        # Mismatch
        delta = x_plant_next - x_twin_next
        mismatch = np.linalg.norm(delta)
        max_mismatch = max(max_mismatch, mismatch)

    return max_mismatch


def estimate_L_J(
    system,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    v: float,
    num_samples: int = 200,
) -> float:
    """
    Estimate the Jacobian Lipschitz constant L_J.

    From Assumption 3: ||[A(k) - A(s), B(k) - B(s)]|| <= L_J * v * |k - s|

    We estimate L_J by sampling pairs (k, s) and computing:
    L_J = max ||[A(k) - A(s), B(k) - B(s)]|| / (v * |k - s|)

    Args:
        system: DiscreteSystem with step() method (plant or twin)
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        v: Trajectory speed (max ||[dx; du]||)
        num_samples: Number of random pairs to sample

    Returns:
        L_J: Estimated Jacobian Lipschitz constant (paper value: 16.94)
    """
    N = len(u_nom)

    # Create function wrapper for Jacobian computation
    def f_discrete(x, u):
        return system.step(x, u)

    # Compute all Jacobians along trajectory
    jacobians = compute_jacobians_along_trajectory(f_discrete, x_nom, u_nom)

    # Sample pairs and estimate L_J
    max_L_J = 0.0
    rng = np.random.default_rng(42)  # Reproducible

    for _ in range(num_samples):
        k = rng.integers(0, N)
        s = rng.integers(0, N)

        if k == s:
            continue

        A_k, B_k = jacobians[k]
        A_s, B_s = jacobians[s]

        diff_norm = jacobian_difference_norm(A_k, B_k, A_s, B_s)
        denominator = v * abs(k - s)

        if denominator > 1e-10:
            L_J_sample = diff_norm / denominator
            max_L_J = max(max_L_J, L_J_sample)

    return max_L_J


def estimate_L_J_exhaustive(
    system,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    v: float,
    step: int = 10,
) -> float:
    """
    Estimate L_J by exhaustively checking pairs with given step size.

    More thorough than random sampling but slower.

    Args:
        system: DiscreteSystem with step() method
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        v: Trajectory speed
        step: Step size for pair enumeration (1 = all pairs)

    Returns:
        L_J: Estimated Jacobian Lipschitz constant
    """
    N = len(u_nom)

    def f_discrete(x, u):
        return system.step(x, u)

    jacobians = compute_jacobians_along_trajectory(f_discrete, x_nom, u_nom)

    max_L_J = 0.0

    for k in range(0, N, step):
        for s in range(k + 1, N, step):
            A_k, B_k = jacobians[k]
            A_s, B_s = jacobians[s]

            diff_norm = jacobian_difference_norm(A_k, B_k, A_s, B_s)
            denominator = v * abs(k - s)

            if denominator > 1e-10:
                L_J_sample = diff_norm / denominator
                max_L_J = max(max_L_J, L_J_sample)

    return max_L_J


def estimate_L_r(
    system,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    radius: float = 0.1,
    num_samples: int = 100,
) -> float:
    """
    Estimate the second-order remainder Lipschitz constant L_r.

    For the Taylor expansion: f(x+eta, u+xi) = f(x,u) + A*eta + B*xi + r(eta, xi)
    We need: ||r|| <= L_r * ||[eta; xi]||^2

    This is estimated by sampling perturbations around nominal points.

    Args:
        system: DiscreteSystem with step() method
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        radius: Maximum perturbation radius for sampling
        num_samples: Number of samples per nominal point

    Returns:
        L_r: Estimated remainder Lipschitz constant (paper value: 0.074)
    """
    N = len(u_nom)
    n = x_nom.shape[1]
    m = u_nom.shape[1]

    def f_discrete(x, u):
        return system.step(x, u)

    max_L_r = 0.0
    rng = np.random.default_rng(42)

    # Sample a subset of nominal points
    sample_indices = rng.choice(N, size=min(50, N), replace=False)

    for k in sample_indices:
        x_k = x_nom[k]
        u_k = u_nom[k]

        # Compute Jacobians at this point
        A_k, B_k = numerical_jacobian(f_discrete, x_k, u_k)

        # Nominal next state
        f_nom = f_discrete(x_k, u_k)

        for _ in range(num_samples):
            # Random perturbation
            eta = rng.uniform(-radius, radius, size=n)
            xi = rng.uniform(-radius, radius, size=m)

            # Perturbed state and input
            x_pert = x_k + eta
            u_pert = u_k + xi

            # Actual next state
            f_pert = f_discrete(x_pert, u_pert)

            # Linear approximation
            f_linear = f_nom + A_k @ eta + B_k @ xi

            # Remainder
            r = f_pert - f_linear

            # Compute L_r estimate
            pert_norm_sq = np.linalg.norm(eta) ** 2 + np.linalg.norm(xi) ** 2
            r_norm = np.linalg.norm(r)

            if pert_norm_sq > 1e-12:
                L_r_sample = r_norm / pert_norm_sq
                max_L_r = max(max_L_r, L_r_sample)

    return max_L_r


def compute_C(L_J: float, v: float) -> float:
    """
    Compute the Jacobian variation bound C = L_J * v.

    From Assumption 3, this bounds the Jacobian variation:
    ||[A(k) - A(s), B(k) - B(s)]|| <= C * |k - s|

    Args:
        L_J: Jacobian Lipschitz constant
        v: Trajectory speed

    Returns:
        C: Jacobian variation bound (paper value: 3.64)
    """
    return L_J * v


def estimate_all_constants(
    plant,
    twin,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    v: float,
) -> dict:
    """
    Estimate all Lipschitz constants needed for funnel synthesis.

    Args:
        plant: Plant DiscreteSystem
        twin: Twin DiscreteSystem
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        v: Trajectory speed

    Returns:
        dict with keys: gamma, L_J, L_r, C, v
    """
    gamma = estimate_gamma(plant, twin, x_nom, u_nom)
    L_J = estimate_L_J(twin, x_nom, u_nom, v)
    L_r = estimate_L_r(twin, x_nom, u_nom)
    C = compute_C(L_J, v)

    return {
        "gamma": gamma,
        "L_J": L_J,
        "L_r": L_r,
        "C": C,
        "v": v,
    }
