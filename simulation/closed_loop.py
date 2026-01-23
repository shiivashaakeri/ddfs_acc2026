"""
Main Closed-Loop Simulation.

Run full closed-loop simulation with the segmented funnel controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.params import INPUT_BOUNDS
from control.segmented_controller import (
    SegmentedFunnelController,
    create_controller_from_params,
)

# Input saturation limits
U_MIN = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
U_MAX = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])


@dataclass
class SimulationResults:
    """Container for simulation results."""

    x_trajectory: np.ndarray  # (N+1, n) plant state trajectory
    u_trajectory: np.ndarray  # (N, m) control inputs
    x_nom: np.ndarray  # (N+1, n) nominal trajectory
    u_nom: np.ndarray  # (N, m) nominal inputs
    eta_trajectory: np.ndarray  # (N+1, n) state deviations
    P_history: List[np.ndarray]  # P matrices per segment
    K_history: List[np.ndarray]  # K matrices per segment
    beta_history: List[float]  # Beta values per segment
    sdp_results: List[dict]  # SDP solver diagnostics
    eta_final_norm: float  # Terminal deviation norm
    max_eta_norm: float  # Maximum deviation norm
    all_sdps_succeeded: bool  # All SDPs solved
    in_funnel_fraction: float  # Fraction of time in funnel


def run_simulation(
    plant,
    twin,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x0_plant: np.ndarray,
    controller: Optional[SegmentedFunnelController] = None,
    verbose: bool = True,
) -> SimulationResults:
    """
    Run full closed-loop simulation with segmented funnel controller.

    Args:
        plant: DiscreteSystem for physical plant
        twin: DiscreteSystem for digital twin
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        x0_plant: (n,) Initial plant state (different from nominal!)
        controller: Optional pre-configured controller
        verbose: Print progress

    Returns:
        SimulationResults with trajectories, funnels, and metrics
    """
    x_nom = np.asarray(x_nom)
    u_nom = np.asarray(u_nom)
    x0_plant = np.asarray(x0_plant)

    N = u_nom.shape[0]
    n = x_nom.shape[1]
    m = u_nom.shape[1]

    # Create controller if not provided
    if controller is None:
        controller = create_controller_from_params(twin, x_nom, u_nom)

    # Storage
    x_trajectory = np.zeros((N + 1, n))
    u_trajectory = np.zeros((N, m))
    eta_trajectory = np.zeros((N + 1, n))

    # Initialize
    x = x0_plant.copy()
    x_trajectory[0] = x
    eta_trajectory[0] = x - x_nom[0]

    if verbose:
        print(f"Starting simulation: N={N}, {controller.num_segments} segments")
        print(f"Initial deviation ||eta(0)|| = {np.linalg.norm(eta_trajectory[0]):.4f}")

    # Track funnel containment
    in_funnel_count = 0

    # Main simulation loop
    for k in range(N):
        # Compute control
        u = controller.compute_control(k, x)

        # Saturate control inputs to prevent numerical issues
        u = np.clip(u, U_MIN, U_MAX)
        u_trajectory[k] = u

        # Apply to plant
        x_next = plant.step(x, u)

        # Check for numerical overflow
        if not np.all(np.isfinite(x_next)):
            if verbose:
                print(f"Warning: State overflow at k={k}, terminating simulation early")
            # Fill remaining trajectory with last valid state
            x_trajectory[k + 1:] = x
            eta_trajectory[k + 1:] = x - x_nom[k + 1:]
            break

        x_trajectory[k + 1] = x_next
        eta_trajectory[k + 1] = x_next - x_nom[k + 1]

        # Check funnel containment
        if controller.in_funnel(k, eta_trajectory[k]):
            in_funnel_count += 1

        # End of segment update
        controller.end_of_segment_update(k)

        # Update state
        x = x_next

        # Progress reporting
        if verbose and (k + 1) % controller.T == 0:
            segment = (k + 1) // controller.T
            eta_norm = np.linalg.norm(eta_trajectory[k + 1])
            print(f"  Segment {segment}/{controller.num_segments} complete, ||eta|| = {eta_norm:.4f}")

    # Compute metrics
    eta_norms = np.linalg.norm(eta_trajectory, axis=1)
    eta_final_norm = eta_norms[-1]
    max_eta_norm = eta_norms.max()
    in_funnel_fraction = in_funnel_count / N

    all_sdps_succeeded = all(r["success"] for r in controller.sdp_results) if controller.sdp_results else True

    if verbose:
        print(f"Simulation complete:")
        print(f"  Terminal ||eta(N)|| = {eta_final_norm:.4e}")
        print(f"  Maximum ||eta|| = {max_eta_norm:.4f}")
        print(f"  In-funnel fraction = {in_funnel_fraction:.2%}")
        print(f"  All SDPs succeeded = {all_sdps_succeeded}")

    return SimulationResults(
        x_trajectory=x_trajectory,
        u_trajectory=u_trajectory,
        x_nom=x_nom,
        u_nom=u_nom,
        eta_trajectory=eta_trajectory,
        P_history=controller.P_history,
        K_history=controller.K_history,
        beta_history=controller.beta_history,
        sdp_results=controller.sdp_results,
        eta_final_norm=eta_final_norm,
        max_eta_norm=max_eta_norm,
        all_sdps_succeeded=all_sdps_succeeded,
        in_funnel_fraction=in_funnel_fraction,
    )


def compute_tracking_metrics(results: SimulationResults) -> dict:
    """
    Compute detailed tracking metrics from simulation results.

    Args:
        results: SimulationResults from run_simulation

    Returns:
        dict with various tracking metrics
    """
    eta = results.eta_trajectory
    eta_norms = np.linalg.norm(eta, axis=1)

    # Per-state metrics
    eta_max_per_state = np.abs(eta).max(axis=0)
    eta_rms_per_state = np.sqrt(np.mean(eta**2, axis=0))

    return {
        "eta_final_norm": results.eta_final_norm,
        "eta_max_norm": eta_norms.max(),
        "eta_mean_norm": eta_norms.mean(),
        "eta_rms_norm": np.sqrt(np.mean(eta_norms**2)),
        "eta_max_per_state": eta_max_per_state,
        "eta_rms_per_state": eta_rms_per_state,
        "in_funnel_fraction": results.in_funnel_fraction,
        "num_segments": len(results.P_history),
        "all_sdps_succeeded": results.all_sdps_succeeded,
    }
