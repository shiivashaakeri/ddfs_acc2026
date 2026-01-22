"""
Trajectory Utility Functions.

Utility functions for trajectory analysis including:
- Computing trajectory speed v for Assumption 3
- Constraint checking
- Deviation computation
"""

from __future__ import annotations

import numpy as np


def compute_trajectory_speed(x_nom: np.ndarray, u_nom: np.ndarray) -> float:
    """
    Compute trajectory speed v for Assumption 3.

    v = max_{j in [0, N-2]} ||[x_nom(j+1) - x_nom(j); u_nom(j+1) - u_nom(j)]||_2

    This measures the maximum rate of change along the nominal trajectory,
    used for bounding Jacobian variation in the data-driven approach.

    Args:
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory

    Returns:
        v: Maximum trajectory speed
    """
    x_nom = np.asarray(x_nom, dtype=float)
    u_nom = np.asarray(u_nom, dtype=float)

    N = len(u_nom) - 1  # Number of segments for computing differences

    if N < 1:
        return 0.0

    max_speed = 0.0
    for j in range(N):
        # State difference
        dx = x_nom[j + 1] - x_nom[j]
        # Input difference
        du = u_nom[j + 1] - u_nom[j]
        # Combined vector
        combined = np.hstack((dx, du))
        speed = np.linalg.norm(combined)
        max_speed = max(max_speed, speed)

    return max_speed


def check_constraints(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    state_bounds: dict,
    input_bounds: dict,
) -> tuple[bool, list[str]]:
    """
    Check if trajectory satisfies state and input constraints.

    Args:
        x_traj: (N+1, n) State trajectory [q1, q2, dq1, dq2]
        u_traj: (N, m) Input trajectory [tau1, tau2]
        state_bounds: Dict with keys 'q1', 'q2', 'dq1', 'dq2',
                      each value is (lower, upper)
        input_bounds: Dict with keys 'tau1', 'tau2',
                      each value is (lower, upper)

    Returns:
        is_valid: True if all constraints satisfied
        violations: List of violation descriptions
    """
    violations = []

    # Check state bounds
    state_keys = ['q1', 'q2', 'dq1', 'dq2']
    for i, key in enumerate(state_keys):
        if key in state_bounds:
            lower, upper = state_bounds[key]
            min_val = np.min(x_traj[:, i])
            max_val = np.max(x_traj[:, i])
            if min_val < lower:
                violations.append(f"{key}: min={min_val:.4f} < {lower}")
            if max_val > upper:
                violations.append(f"{key}: max={max_val:.4f} > {upper}")

    # Check input bounds
    input_keys = ['tau1', 'tau2']
    for i, key in enumerate(input_keys):
        if key in input_bounds:
            lower, upper = input_bounds[key]
            min_val = np.min(u_traj[:, i])
            max_val = np.max(u_traj[:, i])
            if min_val < lower:
                violations.append(f"{key}: min={min_val:.4f} < {lower}")
            if max_val > upper:
                violations.append(f"{key}: max={max_val:.4f} > {upper}")

    is_valid = len(violations) == 0
    return is_valid, violations


def compute_deviations(x_actual: np.ndarray, x_nom: np.ndarray) -> np.ndarray:
    """
    Compute deviation eta = x_actual - x_nom along trajectory.

    Args:
        x_actual: (N+1, n) Actual state trajectory
        x_nom: (N+1, n) Nominal state trajectory

    Returns:
        eta: (N+1, n) Deviation trajectory
    """
    return np.asarray(x_actual, dtype=float) - np.asarray(x_nom, dtype=float)


def compute_max_deviation(x_actual: np.ndarray, x_nom: np.ndarray) -> float:
    """
    Compute maximum deviation norm along trajectory.

    Args:
        x_actual: (N+1, n) Actual state trajectory
        x_nom: (N+1, n) Nominal state trajectory

    Returns:
        max_dev: Maximum ||eta(k)||_2 over all time steps
    """
    eta = compute_deviations(x_actual, x_nom)
    return float(np.max(np.linalg.norm(eta, axis=1)))


def segment_trajectory(
    x_nom: np.ndarray, u_nom: np.ndarray, T: int
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Segment trajectory into chunks of length T for data collection.

    Args:
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        T: Segment length (number of time steps)

    Returns:
        segments: List of (x_segment, u_segment, start_index) tuples
    """
    N = len(u_nom)
    segments = []

    for start in range(0, N, T):
        end = min(start + T, N)
        # State includes one extra point for final state
        x_segment = x_nom[start : end + 1]
        u_segment = u_nom[start:end]
        segments.append((x_segment, u_segment, start))

    return segments
