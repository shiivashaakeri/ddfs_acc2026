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


def generate_nominal_trajectory(
    twin,
    x0: np.ndarray,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    N: int,
    Q: np.ndarray = None,
    R: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nominal trajectory on digital twin using LQR tracking.

    The controller is: u(k) = u_goal - K @ (x(k) - x_goal)
    where K is the discrete LQR gain computed at the goal linearization.

    Args:
        twin: Digital twin DiscreteSystem
        x0: Initial state (n,)
        x_goal: Goal state (n,)
        u_goal: Goal input / equilibrium torques (m,)
        N: Number of time steps
        Q: State cost matrix (n, n). Default: diag(100, 100, 10, 10)
        R: Input cost matrix (m, m). Default: 0.5 * I

    Returns:
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
    """
    x0 = np.asarray(x0, dtype=float)
    x_goal = np.asarray(x_goal, dtype=float)
    u_goal = np.asarray(u_goal, dtype=float)

    n = len(x0)
    m = len(u_goal)

    # Default LQR weights tuned for convergence within torque limits
    if Q is None:
        Q = np.diag([100.0, 100.0, 10.0, 10.0])
    else:
        Q = np.asarray(Q, dtype=float)

    if R is None:
        R = np.eye(m) * 0.5
    else:
        R = np.asarray(R, dtype=float)

    # Step 1: Linearize twin at goal point
    A, B = linearize_at_point(twin, x_goal, u_goal)

    # Step 2: Compute LQR gain
    K = compute_lqr_gain(A, B, Q, R)

    # Step 3: Simulate twin with LQR controller
    x_nom = np.zeros((N + 1, n))
    u_nom = np.zeros((N, m))

    x_nom[0] = x0
    x = x0.copy()

    for k in range(N):
        # LQR control law: u = u_goal - K @ (x - x_goal)
        u = u_goal - K @ (x - x_goal)
        u_nom[k] = u

        # Step the twin
        x = twin.step(x, u)
        x_nom[k + 1] = x

    return x_nom, u_nom
