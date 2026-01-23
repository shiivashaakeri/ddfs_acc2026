"""
Beta (disturbance energy bound) Computation.

Compute beta_i from Eq. (16) in the paper:
beta_i = sum_{k in T_i^D} (C|k-k_i| * ||z(k)|| + gamma + L_r * ||z(k)||^2)^2

where z(k) = [eta(k); xi(k)] is the combined state-input deviation.

This bound captures:
- Jacobian variation error: C|k-k_i| * ||z||
- Plant-twin mismatch: gamma
- Second-order remainder: L_r * ||z||^2
"""

from __future__ import annotations

from typing import Union

import numpy as np

from data_driven.data_collection import SegmentManager


def compute_beta(
    eta_log: np.ndarray,
    xi_log: np.ndarray,
    data_window: range,
    k_i: int,
    C: float,
    gamma: float,
    L_r: float,
) -> float:
    """
    Compute disturbance energy bound beta_i from Eq. (16).

    beta_i = sum_{k in T_i^D} (C|k-k_i| * ||z(k)|| + gamma + L_r * ||z(k)||^2)^2

    Args:
        eta_log: (N+1, n) State deviation log eta(k) = x(k) - x_nom(k)
        xi_log: (N, m) Input deviation log xi(k) = u(k) - u_nom(k)
        data_window: Indices in T_i^D (range object)
        k_i: Start index of segment i
        C: Jacobian variation bound (C = L_J * v)
        gamma: Plant-twin mismatch bound
        L_r: Second-order remainder Lipschitz constant

    Returns:
        beta_i: Disturbance energy bound (scalar)
    """
    beta_i = 0.0

    for k in data_window:
        # Get deviations at time k
        eta_k = eta_log[k]
        xi_k = xi_log[k]

        # Combined deviation z(k) = [eta(k); xi(k)]
        z_k = np.concatenate([eta_k, xi_k])
        norm_z = np.linalg.norm(z_k)

        # Compute bound term from Eq. (16)
        # term = C * |k - k_i| * ||z|| + gamma + L_r * ||z||^2
        term = C * abs(k - k_i) * norm_z + gamma + L_r * norm_z**2

        # Accumulate squared term
        beta_i += term**2

    return beta_i


def compute_beta_for_segment(
    eta_log: np.ndarray,
    xi_log: np.ndarray,
    segment_i: int,
    segment_manager: SegmentManager,
    C: float,
    gamma: float,
    L_r: float,
) -> float:
    """
    Compute beta_i for a specific segment using SegmentManager.

    Convenience wrapper around compute_beta.

    Args:
        eta_log: (N+1, n) State deviation log
        xi_log: (N, m) Input deviation log
        segment_i: Segment index
        segment_manager: SegmentManager instance
        C: Jacobian variation bound
        gamma: Plant-twin mismatch bound
        L_r: Second-order remainder Lipschitz constant

    Returns:
        beta_i: Disturbance energy bound for segment i
    """
    data_window = segment_manager.get_data_window_indices(segment_i)
    k_i = segment_manager.get_segment_start(segment_i)

    return compute_beta(eta_log, xi_log, data_window, k_i, C, gamma, L_r)


def compute_individual_disturbance_bound(
    k: int,
    k_i: int,
    eta_k: np.ndarray,
    xi_k: np.ndarray,
    C: float,
    gamma: float,
    L_r: float,
) -> float:
    """
    Compute individual disturbance bound ||d(k)|| for a single time step.

    From Eq. (15): ||d(k)|| <= C|k-k_i| * ||z(k)|| + gamma + L_r * ||z(k)||^2

    Args:
        k: Time index
        k_i: Segment start index
        eta_k: State deviation at k
        xi_k: Input deviation at k
        C: Jacobian variation bound
        gamma: Plant-twin mismatch bound
        L_r: Second-order remainder constant

    Returns:
        Bound on ||d(k)||
    """
    z_k = np.concatenate([eta_k, xi_k])
    norm_z = np.linalg.norm(z_k)

    return C * abs(k - k_i) * norm_z + gamma + L_r * norm_z**2


def compute_beta_decomposition(
    eta_log: np.ndarray,
    xi_log: np.ndarray,
    data_window: range,
    k_i: int,
    C: float,
    gamma: float,
    L_r: float,
) -> dict:
    """
    Compute beta with detailed breakdown of contributions.

    Useful for analysis and debugging.

    Args:
        eta_log: State deviation log
        xi_log: Input deviation log
        data_window: Data window indices
        k_i: Segment start index
        C, gamma, L_r: Lipschitz constants

    Returns:
        dict with:
            beta: Total beta value
            jacobian_contrib: Contribution from C|k-k_i|*||z||
            mismatch_contrib: Contribution from gamma
            remainder_contrib: Contribution from L_r*||z||^2
            max_z_norm: Maximum ||z|| in data window
            mean_z_norm: Mean ||z|| in data window
    """
    beta = 0.0
    jacobian_contrib = 0.0
    mismatch_contrib = 0.0
    remainder_contrib = 0.0
    z_norms = []

    for k in data_window:
        eta_k = eta_log[k]
        xi_k = xi_log[k]
        z_k = np.concatenate([eta_k, xi_k])
        norm_z = np.linalg.norm(z_k)
        z_norms.append(norm_z)

        jac_term = C * abs(k - k_i) * norm_z
        mis_term = gamma
        rem_term = L_r * norm_z**2

        total_term = jac_term + mis_term + rem_term
        beta += total_term**2

        # Track individual contributions (squared)
        jacobian_contrib += jac_term**2
        mismatch_contrib += mis_term**2
        remainder_contrib += rem_term**2

    return {
        "beta": beta,
        "jacobian_contrib": jacobian_contrib,
        "mismatch_contrib": mismatch_contrib,
        "remainder_contrib": remainder_contrib,
        "max_z_norm": max(z_norms) if z_norms else 0.0,
        "mean_z_norm": np.mean(z_norms) if z_norms else 0.0,
        "num_samples": len(list(data_window)),
    }
