"""
Numerical Jacobian Computation.

Compute Jacobians A(k), B(k) of the discrete-time plant/twin
at nominal points using finite differences.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def numerical_jacobian(
    f_discrete: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobians A = df/dx and B = df/du at (x, u) using central differences.

    For a discrete-time system x_{k+1} = f(x_k, u_k), computes:
    - A = ∂f/∂x evaluated at (x, u)
    - B = ∂f/∂u evaluated at (x, u)

    Args:
        f_discrete: Function x_next = f(x, u) representing discrete dynamics
        x: Linearization point state (n,)
        u: Linearization point input (m,)
        eps: Finite difference step size

    Returns:
        A: (n, n) State Jacobian
        B: (n, m) Input Jacobian
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
        A[:, i] = (f_discrete(x_plus, u) - f_discrete(x_minus, u)) / (2 * eps)

    # Compute B = ∂f/∂u using central differences
    B = np.zeros((n, m))
    for j in range(m):
        u_plus = u.copy()
        u_plus[j] += eps
        u_minus = u.copy()
        u_minus[j] -= eps
        B[:, j] = (f_discrete(x, u_plus) - f_discrete(x, u_minus)) / (2 * eps)

    return A, B


def compute_jacobians_along_trajectory(
    f_discrete: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    eps: float = 1e-6,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute Jacobians (A(k), B(k)) along the entire nominal trajectory.

    Args:
        f_discrete: Function x_next = f(x, u) representing discrete dynamics
        x_nom: (N+1, n) Nominal state trajectory
        u_nom: (N, m) Nominal input trajectory
        eps: Finite difference step size

    Returns:
        jacobians: List of (A(k), B(k)) tuples for k = 0, ..., N-1
    """
    N = len(u_nom)
    jacobians = []

    for k in range(N):
        A_k, B_k = numerical_jacobian(f_discrete, x_nom[k], u_nom[k], eps)
        jacobians.append((A_k, B_k))

    return jacobians


def jacobian_matrix_norm(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the induced 2-norm of the combined Jacobian matrix [A, B].

    This is used for Lipschitz constant estimation.

    Args:
        A: (n, n) State Jacobian
        B: (n, m) Input Jacobian

    Returns:
        norm: ||[A, B]||_2 (spectral norm)
    """
    AB = np.hstack([A, B])
    return np.linalg.norm(AB, ord=2)


def jacobian_difference_norm(
    A1: np.ndarray, B1: np.ndarray, A2: np.ndarray, B2: np.ndarray
) -> float:
    """
    Compute the norm of the difference between two Jacobian pairs.

    ||[A1 - A2, B1 - B2]||_2

    Args:
        A1, B1: First Jacobian pair
        A2, B2: Second Jacobian pair

    Returns:
        norm: ||[A1-A2, B1-B2]||_2
    """
    dA = A1 - A2
    dB = B1 - B2
    dAB = np.hstack([dA, dB])
    return np.linalg.norm(dAB, ord=2)
