"""
Baseline Comparison (Twin-only Controller).

Run baseline simulation where the twin's nominal controller
is directly applied to the plant without data-driven adaptation.

This represents the "model-based" approach that assumes the twin
is a perfect model of the plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg

from config.params import INPUT_BOUNDS
from trajectory.nominal_planner import linearize_at_point, compute_lqr_gain

# Input saturation limits
U_MIN = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
U_MAX = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])


@dataclass
class BaselineResults:
    """Container for baseline simulation results."""

    x_trajectory: np.ndarray  # (N+1, n) plant state trajectory
    u_trajectory: np.ndarray  # (N, m) control inputs
    x_nom: np.ndarray  # (N+1, n) nominal trajectory
    u_nom: np.ndarray  # (N, m) nominal inputs
    eta_trajectory: np.ndarray  # (N+1, n) state deviations
    K: np.ndarray  # (m, n) feedback gain used
    eta_final_norm: float  # Terminal deviation norm
    max_eta_norm: float  # Maximum deviation norm


def design_twin_controller(
    twin,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    Q: np.ndarray = None,
    R: np.ndarray = None,
) -> np.ndarray:
    """
    Design LQR controller based on digital twin linearized at goal.

    Args:
        twin: Digital twin DiscreteSystem
        x_goal: Goal state
        u_goal: Goal input
        Q: State cost matrix
        R: Input cost matrix

    Returns:
        K: (m, n) LQR feedback gain
    """
    n = len(x_goal)
    m = len(u_goal)

    if Q is None:
        Q = np.diag([500.0, 500.0, 100.0, 100.0])
    if R is None:
        R = np.eye(m) * 0.1

    # Linearize twin at goal
    A, B = linearize_at_point(twin, x_goal, u_goal)

    # Compute LQR gain
    K = compute_lqr_gain(A, B, Q, R)

    return K


def run_baseline(
    plant,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x0_plant: np.ndarray,
    K: np.ndarray,
    verbose: bool = True,
) -> BaselineResults:
    """
    Run baseline simulation with fixed twin-designed controller on plant.

    This applies the twin-designed controller directly to the plant,
    representing a model-based approach without data-driven adaptation.

    u(k) = u_nom(k) + K @ (x(k) - x_nom(k))

    Args:
        plant: DiscreteSystem for physical plant
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        x0_plant: (n,) Initial plant state
        K: (m, n) Feedback gain (designed for twin)
        verbose: Print progress

    Returns:
        BaselineResults with trajectory and metrics
    """
    x_nom = np.asarray(x_nom)
    u_nom = np.asarray(u_nom)
    x0_plant = np.asarray(x0_plant)
    K = np.asarray(K)

    N = u_nom.shape[0]
    n = x_nom.shape[1]
    m = u_nom.shape[1]

    # Storage
    x_trajectory = np.zeros((N + 1, n))
    u_trajectory = np.zeros((N, m))
    eta_trajectory = np.zeros((N + 1, n))

    # Initialize
    x = x0_plant.copy()
    x_trajectory[0] = x
    eta_trajectory[0] = x - x_nom[0]

    if verbose:
        print(f"Running baseline simulation: N={N}")
        print(f"Initial deviation ||eta(0)|| = {np.linalg.norm(eta_trajectory[0]):.4f}")

    # Main simulation loop
    for k in range(N):
        # Compute deviation
        eta_k = x - x_nom[k]

        # Feedback control (no excitation, fixed K)
        u = u_nom[k] + K @ eta_k

        # Saturate control inputs
        u = np.clip(u, U_MIN, U_MAX)
        u_trajectory[k] = u

        # Apply to plant
        x_next = plant.step(x, u)

        # Check for numerical overflow
        if not np.all(np.isfinite(x_next)):
            if verbose:
                print(f"Warning: State overflow at k={k}, terminating simulation early")
            x_trajectory[k + 1:] = x
            eta_trajectory[k + 1:] = x - x_nom[k + 1:]
            break

        x_trajectory[k + 1] = x_next
        eta_trajectory[k + 1] = x_next - x_nom[k + 1]

        # Update
        x = x_next

    # Compute metrics
    eta_norms = np.linalg.norm(eta_trajectory, axis=1)
    eta_final_norm = eta_norms[-1]
    max_eta_norm = eta_norms.max()

    if verbose:
        print(f"Baseline complete:")
        print(f"  Terminal ||eta(N)|| = {eta_final_norm:.4e}")
        print(f"  Maximum ||eta|| = {max_eta_norm:.4f}")

    return BaselineResults(
        x_trajectory=x_trajectory,
        u_trajectory=u_trajectory,
        x_nom=x_nom,
        u_nom=u_nom,
        eta_trajectory=eta_trajectory,
        K=K,
        eta_final_norm=eta_final_norm,
        max_eta_norm=max_eta_norm,
    )


def compare_methods(
    proposed_results,
    baseline_results,
) -> dict:
    """
    Compare proposed method with baseline.

    Args:
        proposed_results: SimulationResults from proposed method
        baseline_results: BaselineResults from baseline

    Returns:
        dict with comparison metrics
    """
    proposed_eta = np.linalg.norm(proposed_results.eta_trajectory, axis=1)
    baseline_eta = np.linalg.norm(baseline_results.eta_trajectory, axis=1)

    return {
        "proposed_final_norm": proposed_results.eta_final_norm,
        "baseline_final_norm": baseline_results.eta_final_norm,
        "improvement_final": baseline_results.eta_final_norm / (proposed_results.eta_final_norm + 1e-10),
        "proposed_max_norm": proposed_eta.max(),
        "baseline_max_norm": baseline_eta.max(),
        "improvement_max": baseline_eta.max() / (proposed_eta.max() + 1e-10),
        "proposed_mean_norm": proposed_eta.mean(),
        "baseline_mean_norm": baseline_eta.mean(),
        "improvement_mean": baseline_eta.mean() / (proposed_eta.mean() + 1e-10),
    }
