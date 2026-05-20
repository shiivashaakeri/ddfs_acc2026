"""Finite-difference Jacobians of the discrete-time plant map.

The paper linearizes the discrete-time deviation dynamics (Eq. 4) along the
nominal trajectory. We expose A(k) = df_d/dx and B(k) = df_d/du where
f_d(x,u) is the RK4 one-step map of the continuous plant dynamics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def jacobians_discrete(
    step_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (A, B) = (df_d/dx, df_d/du) by central differences.

    step_fn(x, u) is the one-step discrete map (e.g. DiscreteSystem.step).
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    n, m = x.size, u.size

    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        A[:, j] = (step_fn(x + e, u) - step_fn(x - e, u)) / (2.0 * eps)

    B = np.zeros((n, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = eps
        B[:, j] = (step_fn(x, u + e) - step_fn(x, u - e)) / (2.0 * eps)

    return A, B


def jacobians_along_trajectory(
    step_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack Jacobians along a nominal trajectory.

    x_nom : (N+1, n), u_nom : (N, m)
    Returns A_seq : (N, n, n), B_seq : (N, n, m).
    """
    N = u_nom.shape[0]
    A_list, B_list = [], []
    for k in range(N):
        A_k, B_k = jacobians_discrete(step_fn, x_nom[k], u_nom[k], eps=eps)
        A_list.append(A_k)
        B_list.append(B_k)
    return np.stack(A_list), np.stack(B_list)
