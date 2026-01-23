"""
Nominal Trajectory Planning.

Generate LQR-based nominal trajectory on the digital twin.

Approach:
1. Linearize twin at goal (x_goal, u_goal)
2. Compute infinite-horizon LQR gain K_lqr
3. Simulate twin from x_nom_0 using u = u_goal + K_lqr(x - x_goal)
4. Store {x_nom(k)}_{k=0}^N and {u_nom(k)}_{k=0}^{N-1}
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg


def linearize_at_point(
    system, x: np.ndarray, u: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearize discrete-time system at (x, u) using finite differences.

    Args:
        system: DiscreteSystem with step(x, u) method
        x: State linearization point (n,)
        u: Input linearization point (m,)
        eps: Finite difference step size

    Returns:
        A: (n, n) State Jacobian ∂f/∂x
        B: (n, m) Input Jacobian ∂f/∂u
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    n = len(x)
    m = len(u)

    # Compute A = ∂f/∂x using central differences
    A = np.zeros((n, n))
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        A[:, i] = (system.step(x_plus, u) - system.step(x_minus, u)) / (2 * eps)

    # Compute B = ∂f/∂u using central differences
    B = np.zeros((n, m))
    for j in range(m):
        u_plus = u.copy()
        u_plus[j] += eps
        u_minus = u.copy()
        u_minus[j] -= eps
        B[:, j] = (system.step(x, u_plus) - system.step(x, u_minus)) / (2 * eps)

    return A, B


def compute_lqr_gain(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """
    Compute discrete-time LQR gain.

    Solves the discrete-time algebraic Riccati equation and returns
    the optimal feedback gain K such that u = -K @ (x - x_goal).

    Args:
        A: (n, n) State matrix
        B: (n, m) Input matrix
        Q: (n, n) State cost matrix (positive semi-definite)
        R: (m, m) Input cost matrix (positive definite)

    Returns:
        K: (m, n) LQR feedback gain
    """
    # Solve discrete-time algebraic Riccati equation using scipy
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    # Compute LQR gain: K = (R + B'PB)^{-1} B'PA
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def _minimum_jerk_trajectory(x0: np.ndarray, x_goal: np.ndarray, N: int) -> np.ndarray:
    """
    Generate minimum-jerk reference trajectory from x0 to x_goal.

    Uses 5th-order polynomial: s(t) = 10t^3 - 15t^4 + 6t^5
    This ensures smooth acceleration/deceleration with zero velocity
    and acceleration at endpoints.

    Args:
        x0: Initial state (n,)
        x_goal: Goal state (n,)
        N: Number of time steps

    Returns:
        x_ref: (N+1, n) Reference trajectory
    """
    t = np.linspace(0, 1, N + 1)
    # Minimum jerk profile: s(t) = 10t^3 - 15t^4 + 6t^5
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    # Interpolate states
    x_ref = np.outer(1 - s, x0) + np.outer(s, x_goal)
    return x_ref


def generate_nominal_trajectory(
    twin,
    x0: np.ndarray,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    N: int,
    Q: np.ndarray = None,
    R: np.ndarray = None,
    arm=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nominal trajectory on digital twin using two-phase LQR tracking.

    Phase 1 (motion): Time-varying reference tracking with minimum-jerk profile
    Phase 2 (settling): Fixed-goal LQR to ensure convergence

    This achieves v ≈ 0.215 to match paper's Assumption 3, while ensuring
    convergence to x_goal with error < 0.001.

    Args:
        twin: Digital twin DiscreteSystem
        x0: Initial state (n,)
        x_goal: Goal state (n,)
        u_goal: Goal input / equilibrium torques (m,)
        N: Number of time steps (default 600)
        Q: State cost matrix (n, n) for settling phase
        R: Input cost matrix (m, m) for settling phase
        arm: RobotArm2DoF instance for gravity compensation

    Returns:
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
    """
    x0 = np.asarray(x0, dtype=float)
    x_goal = np.asarray(x_goal, dtype=float)
    u_goal = np.asarray(u_goal, dtype=float)

    n = len(x0)
    m = len(u_goal)

    # Import arm if not provided
    if arm is None:
        from dynamics.robot_arm import RobotArm2DoF
        from config.params import TWIN_PARAMS
        arm = RobotArm2DoF(**TWIN_PARAMS)

    # Default LQR weights
    if Q is None:
        Q = np.diag([100.0, 100.0, 10.0, 10.0])
    else:
        Q = np.asarray(Q, dtype=float)

    if R is None:
        R = np.eye(m) * 0.5
    else:
        R = np.asarray(R, dtype=float)

    # Two-phase approach tuned for v ≈ 0.215
    # Phase 1: 350 steps (3.5s) for motion with reference tracking
    # Phase 2: remaining steps for settling at goal
    N1 = min(350, N)  # Motion phase
    N2 = N - N1       # Settling phase

    # Phase 1: Motion with reference tracking
    x_ref = _minimum_jerk_trajectory(x0, x_goal, N1)
    Q_motion = np.diag([50.0, 50.0, 5.0, 5.0])
    R_motion = np.eye(m) * 0.5

    x_nom = np.zeros((N + 1, n))
    u_nom = np.zeros((N, m))
    x_nom[0] = x0
    x = x0.copy()

    for k in range(N1):
        # Gravity feedforward at reference
        u_ref_k = arm.gravity_vector(x_ref[k, :2])
        # Linearize at reference point
        A, B = linearize_at_point(twin, x_ref[k], u_ref_k)
        K = compute_lqr_gain(A, B, Q_motion, R_motion)
        # Track reference
        u = u_ref_k - K @ (x - x_ref[k])
        u_nom[k] = u
        x = twin.step(x, u)
        x_nom[k + 1] = x

    # Phase 2: Settling at goal with fixed-point LQR
    if N2 > 0:
        A_goal, B_goal = linearize_at_point(twin, x_goal, u_goal)
        K_goal = compute_lqr_gain(A_goal, B_goal, Q, R)

        for k in range(N1, N):
            u = u_goal - K_goal @ (x - x_goal)
            u_nom[k] = u
            x = twin.step(x, u)
            x_nom[k + 1] = x

    return x_nom, u_nom
