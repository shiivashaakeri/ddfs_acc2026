"""
Ellipsoid Bounds Computation (MVIE).

Compute P_min(k), R_max(k) via Maximum Volume Inscribed Ellipsoid (Remark 2).

For box constraints, the MVIE is a diagonal ellipsoid. Given a box centered
at x_nom with half-widths h_i = min(x_nom_i - x_lower_i, x_upper_i - x_nom_i):
- P_min = diag(1/h^2) for state constraints
- R_max = diag(h^2) for input constraints

Paper values at k=0:
    P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14)
    R_max^{1/2}(0) = diag(28.56, 37.62)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def compute_half_widths(
    nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Compute half-widths for MVIE of a box constraint.

    The half-width in each dimension is the minimum distance from nominal
    to either bound, ensuring the ellipsoid fits inside the box.

    Args:
        nominal: (d,) nominal values (center of ellipsoid)
        lower: (d,) lower bounds
        upper: (d,) upper bounds

    Returns:
        half_widths: (d,) half-widths in each dimension
    """
    dist_to_lower = nominal - lower
    dist_to_upper = upper - nominal

    # Handle cases where nominal is outside bounds (shouldn't happen normally)
    if np.any(dist_to_lower < 0) or np.any(dist_to_upper < 0):
        raise ValueError(
            f"Nominal point outside bounds: "
            f"lower={lower}, nominal={nominal}, upper={upper}"
        )

    half_widths = np.minimum(dist_to_lower, dist_to_upper)

    # Ensure no zero half-widths (would cause division by zero)
    half_widths = np.maximum(half_widths, 1e-10)

    return half_widths


def compute_mvie_state(
    x_nom_k: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> np.ndarray:
    """
    Compute P_min(k) for state constraints via MVIE.

    The ellipsoid is defined as {eta : eta' * P * eta <= 1}.
    For a box with half-widths h, P_min = diag(1/h^2).

    This is the smallest P such that the unit ellipsoid fits inside
    the state constraint box centered at x_nom_k.

    Args:
        x_nom_k: (n,) nominal state at time k
        x_lower: (n,) lower state bounds
        x_upper: (n,) upper state bounds

    Returns:
        P_min_k: (n, n) positive definite diagonal matrix
    """
    half_widths = compute_half_widths(x_nom_k, x_lower, x_upper)

    # P_min = diag(1/h^2)
    # This ensures eta' * P_min * eta <= 1 implies |eta_i| <= h_i
    P_min_k = np.diag(1.0 / half_widths**2)

    return P_min_k


def compute_mvie_input(
    u_nom_k: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
) -> np.ndarray:
    """
    Compute R_max(k) for input constraints via MVIE.

    The ellipsoid is defined as {xi : xi' * R^{-1} * xi <= 1}.
    For a box with half-widths h, R_max = diag(h^2).

    This is the largest R such that the unit ellipsoid fits inside
    the input constraint box centered at u_nom_k.

    Args:
        u_nom_k: (m,) nominal input at time k
        u_lower: (m,) lower input bounds
        u_upper: (m,) upper input bounds

    Returns:
        R_max_k: (m, m) positive definite diagonal matrix
    """
    half_widths = compute_half_widths(u_nom_k, u_lower, u_upper)

    # R_max = diag(h^2)
    # This ensures xi' * R_max^{-1} * xi <= 1 implies |xi_i| <= h_i
    R_max_k = np.diag(half_widths**2)

    return R_max_k


def compute_P_min_sqrt(
    x_nom_k: np.ndarray,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> np.ndarray:
    """
    Compute P_min^{1/2}(k) for state constraints.

    This is the diagonal square root: P_min^{1/2} = diag(1/h).

    Args:
        x_nom_k: (n,) nominal state at time k
        x_lower: (n,) lower state bounds
        x_upper: (n,) upper state bounds

    Returns:
        P_min_sqrt_k: (n, n) diagonal matrix (P_min^{1/2})
    """
    half_widths = compute_half_widths(x_nom_k, x_lower, x_upper)
    return np.diag(1.0 / half_widths)


def compute_R_max_sqrt(
    u_nom_k: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
) -> np.ndarray:
    """
    Compute R_max^{1/2}(k) for input constraints.

    This is the diagonal square root: R_max^{1/2} = diag(h).

    Args:
        u_nom_k: (m,) nominal input at time k
        u_lower: (m,) lower input bounds
        u_upper: (m,) upper input bounds

    Returns:
        R_max_sqrt_k: (m, m) diagonal matrix (R_max^{1/2})
    """
    half_widths = compute_half_widths(u_nom_k, u_lower, u_upper)
    return np.diag(half_widths)


def compute_segment_bounds(
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    segment_indices: range,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    method: str = "min",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P_min_i, R_max_i for segment i.

    For the segment, we need bounds that hold for all k in the segment.
    We can either:
    - "min": Take element-wise minimum of half-widths (most conservative)
    - "start": Use bounds at segment start
    - "end": Use bounds at segment end

    Args:
        x_nom: (N+1, n) nominal state trajectory
        u_nom: (N, m) nominal input trajectory
        segment_indices: range object for segment indices
        x_lower: (n,) lower state bounds
        x_upper: (n,) upper state bounds
        u_lower: (m,) lower input bounds
        u_upper: (m,) upper input bounds
        method: "min", "start", or "end"

    Returns:
        P_min_i: (n, n) state constraint ellipsoid
        R_max_i: (m, m) input constraint ellipsoid
    """
    n = x_nom.shape[1]
    m = u_nom.shape[1]

    if method == "start":
        k = segment_indices.start
        P_min_i = compute_mvie_state(x_nom[k], x_lower, x_upper)
        R_max_i = compute_mvie_input(u_nom[k], u_lower, u_upper)

    elif method == "end":
        k = segment_indices.stop - 1
        P_min_i = compute_mvie_state(x_nom[k], x_lower, x_upper)
        # For input, use k (last valid input index in segment)
        k_u = min(k, u_nom.shape[0] - 1)
        R_max_i = compute_mvie_input(u_nom[k_u], u_lower, u_upper)

    elif method == "min":
        # Conservative: use minimum half-widths across segment
        min_x_half_widths = np.full(n, np.inf)
        min_u_half_widths = np.full(m, np.inf)

        for k in segment_indices:
            x_hw = compute_half_widths(x_nom[k], x_lower, x_upper)
            min_x_half_widths = np.minimum(min_x_half_widths, x_hw)

            if k < u_nom.shape[0]:
                u_hw = compute_half_widths(u_nom[k], u_lower, u_upper)
                min_u_half_widths = np.minimum(min_u_half_widths, u_hw)

        P_min_i = np.diag(1.0 / min_x_half_widths**2)
        R_max_i = np.diag(min_u_half_widths**2)

    else:
        raise ValueError(f"Unknown method: {method}")

    return P_min_i, R_max_i


def compute_all_segment_bounds(
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    T: int,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    u_lower: np.ndarray,
    u_upper: np.ndarray,
    method: str = "start",
) -> Tuple[list, list]:
    """
    Compute P_min_i, R_max_i for all segments.

    Args:
        x_nom: (N+1, n) nominal state trajectory
        u_nom: (N, m) nominal input trajectory
        T: segment length
        x_lower, x_upper: state bounds (n,)
        u_lower, u_upper: input bounds (m,)
        method: "min", "start", or "end"

    Returns:
        P_min_list: list of (n, n) matrices for each segment
        R_max_list: list of (m, m) matrices for each segment
    """
    N = u_nom.shape[0]
    num_segments = N // T

    P_min_list = []
    R_max_list = []

    for i in range(num_segments):
        segment_indices = range(i * T, (i + 1) * T)
        P_min_i, R_max_i = compute_segment_bounds(
            x_nom, u_nom, segment_indices,
            x_lower, x_upper, u_lower, u_upper,
            method=method
        )
        P_min_list.append(P_min_i)
        R_max_list.append(R_max_i)

    return P_min_list, R_max_list


def get_bounds_from_config() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load state and input bounds from config.

    Returns:
        x_lower: (4,) lower state bounds [q1, q2, dq1, dq2]
        x_upper: (4,) upper state bounds
        u_lower: (2,) lower input bounds [tau1, tau2]
        u_upper: (2,) upper input bounds
    """
    from config.params import STATE_BOUNDS, INPUT_BOUNDS

    x_lower = np.array([
        STATE_BOUNDS["q1"][0],
        STATE_BOUNDS["q2"][0],
        STATE_BOUNDS["dq1"][0],
        STATE_BOUNDS["dq2"][0],
    ])

    x_upper = np.array([
        STATE_BOUNDS["q1"][1],
        STATE_BOUNDS["q2"][1],
        STATE_BOUNDS["dq1"][1],
        STATE_BOUNDS["dq2"][1],
    ])

    u_lower = np.array([
        INPUT_BOUNDS["tau1"][0],
        INPUT_BOUNDS["tau2"][0],
    ])

    u_upper = np.array([
        INPUT_BOUNDS["tau1"][1],
        INPUT_BOUNDS["tau2"][1],
    ])

    return x_lower, x_upper, u_lower, u_upper


def verify_paper_values_k0() -> dict:
    """
    Verify MVIE computation matches paper values at k=0.

    Paper values:
        P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14)
        R_max^{1/2}(0) = diag(28.56, 37.62)

    Returns:
        dict with computed values and comparison
    """
    from config.params import INITIAL_STATES

    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()

    # Compute state bounds
    P_min_sqrt_0 = compute_P_min_sqrt(x_nom_0, x_lower, x_upper)
    computed_P_diag = np.diag(P_min_sqrt_0)
    paper_P_diag = np.array([0.19, 0.12, 0.12, 0.14])

    # For R_max, we need to estimate u_nom_0
    # Paper: R_max^{1/2}(0) = diag(28.56, 37.62)
    # This implies u_nom_0 ≈ [11.4, 2.4] to match the half-widths
    # We'll compute what u_nom_0 would need to be
    u_nom_0_inferred = np.array([
        u_upper[0] - 28.56,  # tau1 half-width = 28.56
        u_upper[1] - 37.62,  # tau2 half-width = 37.62
    ])

    return {
        "x_nom_0": x_nom_0,
        "computed_P_min_sqrt_diag": computed_P_diag,
        "paper_P_min_sqrt_diag": paper_P_diag,
        "P_min_match": np.allclose(computed_P_diag, paper_P_diag, atol=0.02),
        "u_nom_0_inferred": u_nom_0_inferred,
        "paper_R_max_sqrt_diag": np.array([28.56, 37.62]),
    }
