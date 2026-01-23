"""
LMI Matrix Construction.

Build S, N1_tilde, N2_tilde matrices for the main LMI (Theorem 1).

The main LMI condition is: S + Ñ₁ + Ñ₂ ≺ 0

Block structure for all matrices (20x20):
    Blocks: [n, n, m, n, m, n] = [4, 4, 2, 4, 2, 4]
    Row/col indices: [0, 4, 8, 10, 14, 16, 20]

Variables:
    - P: (n, n) funnel shape matrix (positive definite)
    - L: (m, n) feedback gain parameter
    - nu: scalar for S-procedure
    - alpha: contraction rate (typically 0.98)

Data-dependent terms:
    - H, H_plus, Xi: data matrices from segment
    - beta: disturbance energy bound
    - C, T_tilde: Jacobian variation parameters
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np

# Block dimensions for 2-DoF arm: n=4, m=2
# Blocks: [n, n, m, n, m, n] = [4, 4, 2, 4, 2, 4]
# Total: 4 + 4 + 2 + 4 + 2 + 4 = 20


def get_block_indices(n: int, m: int) -> list:
    """
    Compute block boundary indices for the 6-block structure.

    Blocks: [n, n, m, n, m, n]

    Args:
        n: State dimension (4)
        m: Input dimension (2)

    Returns:
        List of 7 indices: [0, n, 2n, 2n+m, 3n+m, 3n+2m, 4n+2m]
    """
    return [0, n, 2*n, 2*n+m, 3*n+m, 3*n+2*m, 4*n+2*m]


def get_total_dimension(n: int, m: int) -> int:
    """Get total LMI dimension: 4n + 2m."""
    return 4*n + 2*m


def build_S_matrix(
    P: np.ndarray,
    L: np.ndarray,
    nu: float,
    alpha: float,
    n: int,
    m: int,
) -> np.ndarray:
    """
    Build the 6-block S matrix for the main LMI.

    S(P, L, nu) encodes the funnel dynamics constraint.
    For CVXPY, P and L are decision variables.

    Block structure (symmetric):
        [alpha*P - nu*I,    0,       0,       0,       0,      0   ]
        [     0,          -P,     -L^T,     -P,     -L^T,      0   ]
        [     0,          -L,       0,      -L,       0,       L   ]
        [     0,          -P,     -L^T,     -P,     -L^T,      0   ]
        [     0,          -L,       0,      -L,       0,       L   ]
        [     0,           0,      L^T,      0,      L^T,      P   ]

    Args:
        P: (n, n) Funnel shape matrix
        L: (m, n) Feedback parameter (K = L @ P^{-1})
        nu: Scalar S-procedure multiplier
        alpha: Contraction rate (e.g., 0.98)
        n: State dimension
        m: Input dimension

    Returns:
        S: (4n+2m, 4n+2m) symmetric matrix
    """
    dim = get_total_dimension(n, m)
    idx = get_block_indices(n, m)

    # Initialize S matrix
    S = np.zeros((dim, dim))

    # Block (0,0): alpha*P - nu*I
    S[idx[0]:idx[1], idx[0]:idx[1]] = alpha * P - nu * np.eye(n)

    # Block (1,1): -P
    S[idx[1]:idx[2], idx[1]:idx[2]] = -P

    # Block (1,2) and (2,1): -L^T and -L
    S[idx[1]:idx[2], idx[2]:idx[3]] = -L.T
    S[idx[2]:idx[3], idx[1]:idx[2]] = -L

    # Block (1,3) and (3,1): -P
    S[idx[1]:idx[2], idx[3]:idx[4]] = -P
    S[idx[3]:idx[4], idx[1]:idx[2]] = -P

    # Block (1,4) and (4,1): -L^T and -L
    S[idx[1]:idx[2], idx[4]:idx[5]] = -L.T
    S[idx[4]:idx[5], idx[1]:idx[2]] = -L

    # Block (2,3) and (3,2): -L and -L^T
    S[idx[2]:idx[3], idx[3]:idx[4]] = -L
    S[idx[3]:idx[4], idx[2]:idx[3]] = -L.T

    # Block (2,5) and (5,2): L and L^T
    S[idx[2]:idx[3], idx[5]:idx[6]] = L
    S[idx[5]:idx[6], idx[2]:idx[3]] = L.T

    # Block (3,3): -P
    S[idx[3]:idx[4], idx[3]:idx[4]] = -P

    # Block (3,4) and (4,3): -L^T and -L
    S[idx[3]:idx[4], idx[4]:idx[5]] = -L.T
    S[idx[4]:idx[5], idx[3]:idx[4]] = -L

    # Block (4,5) and (5,4): L and L^T
    S[idx[4]:idx[5], idx[5]:idx[6]] = L
    S[idx[5]:idx[6], idx[4]:idx[5]] = L.T

    # Block (5,5): P
    S[idx[5]:idx[6], idx[5]:idx[6]] = P

    return S


def build_N1_tilde(
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    beta: float,
    n: int,
    m: int,
) -> np.ndarray:
    """
    Build Ñ₁ matrix (data constraint from Definition 2).

    Ñ₁ captures the uncertainty from the data-driven characterization
    of admissible (A, B) pairs.

    Structure: Ñ₁ = S_W^T * M_W * S_W embedded in 6-block format

    Where:
        S_W selects [eta(k); A*eta + B*xi; xi] from the full vector
        M_W encodes the data constraint with beta

    The data matrices are:
        H: (n, L) state deviations
        H_plus: (n, L) next-state deviations
        Xi: (m, L) input deviations

    Args:
        H: (n, L) State deviation matrix
        H_plus: (n, L) Next-state deviation matrix
        Xi: (m, L) Input deviation matrix
        beta: Disturbance energy bound
        n: State dimension
        m: Input dimension

    Returns:
        N1_tilde: (4n+2m, 4n+2m) symmetric matrix
    """
    dim = get_total_dimension(n, m)
    idx = get_block_indices(n, m)
    L_data = H.shape[1]  # Data window length

    # Build the data-dependent constraint matrix
    # The key identity: for all (A,B) in Sigma_i,
    # [H; H_plus; Xi] captures the admissible dynamics

    # Stack data matrices: W = [H; Xi] (n+m, L)
    W = np.vstack([H, Xi])  # (n+m, L)

    # Gramian: W @ W^T
    WWT = W @ W.T  # (n+m, n+m)

    # H_plus contribution
    H_plus_HT = H_plus @ H.T  # (n, n)
    H_plus_XiT = H_plus @ Xi.T  # (n, m)
    H_plus_H_plusT = H_plus @ H_plus.T  # (n, n)

    # Build Ñ₁ in block form
    # The constraint encodes: for disturbances bounded by beta,
    # the next state H_plus = A*H + B*Xi + D where ||D||_F^2 <= beta

    N1_tilde = np.zeros((dim, dim))

    # Block structure encodes the S-procedure for data constraint
    # Using nu as multiplier (absorbed into S matrix)

    # The key terms involve cross-products of H, H_plus, Xi
    # Blocks (1,1), (1,3), (3,1), (3,3) involve H @ H^T
    HHT = H @ H.T  # (n, n)
    XiXiT = Xi @ Xi.T  # (m, m)
    HXiT = H @ Xi.T  # (n, m)

    # Block (1,1): contribution from H @ H^T
    N1_tilde[idx[1]:idx[2], idx[1]:idx[2]] = HHT

    # Block (1,2) and (2,1): H @ Xi^T
    N1_tilde[idx[1]:idx[2], idx[2]:idx[3]] = HXiT
    N1_tilde[idx[2]:idx[3], idx[1]:idx[2]] = HXiT.T

    # Block (2,2): Xi @ Xi^T
    N1_tilde[idx[2]:idx[3], idx[2]:idx[3]] = XiXiT

    # Blocks involving H_plus (next state predictions)
    # Block (1,5) and (5,1): -H @ H_plus^T
    N1_tilde[idx[1]:idx[2], idx[5]:idx[6]] = -H @ H_plus.T
    N1_tilde[idx[5]:idx[6], idx[1]:idx[2]] = -H_plus @ H.T

    # Block (2,5) and (5,2): -Xi @ H_plus^T
    N1_tilde[idx[2]:idx[3], idx[5]:idx[6]] = -Xi @ H_plus.T
    N1_tilde[idx[5]:idx[6], idx[2]:idx[3]] = -H_plus @ Xi.T

    # Block (5,5): H_plus @ H_plus^T + beta * I
    N1_tilde[idx[5]:idx[6], idx[5]:idx[6]] = H_plus_H_plusT + beta * np.eye(n)

    return N1_tilde


def build_N2_tilde(
    C: float,
    T_tilde: int,
    n: int,
    m: int,
) -> np.ndarray:
    """
    Build Ñ₂ matrix (Jacobian variation constraint from Definition 3).

    Ñ₂ captures the uncertainty from time-varying Jacobians.
    The bound is C² * T̃² where C = L_J * v is the Jacobian variation rate.

    Structure: Ñ₂ encodes ||A(k) - A(k_i)||² <= C² * |k - k_i|²

    Args:
        C: Jacobian variation rate (C = L_J * v)
        T_tilde: Maximum index difference (2T - L - 1)
        n: State dimension
        m: Input dimension

    Returns:
        N2_tilde: (4n+2m, 4n+2m) symmetric matrix
    """
    dim = get_total_dimension(n, m)
    idx = get_block_indices(n, m)

    # C² T̃² bound
    C2_T2 = C**2 * T_tilde**2

    N2_tilde = np.zeros((dim, dim))

    # The Jacobian variation adds uncertainty to blocks (3,3), (3,4), (4,3), (4,4)
    # These blocks correspond to the "uncertain" dynamics copies

    # Block (3,3): C² T̃² * I_n (state uncertainty)
    N2_tilde[idx[3]:idx[4], idx[3]:idx[4]] = C2_T2 * np.eye(n)

    # Block (4,4): C² T̃² * I_m (input uncertainty through B matrix)
    N2_tilde[idx[4]:idx[5], idx[4]:idx[5]] = C2_T2 * np.eye(m)

    return N2_tilde


def build_full_lmi_matrix(
    P: np.ndarray,
    L: np.ndarray,
    nu: float,
    alpha: float,
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    beta: float,
    C: float,
    T_tilde: int,
    n: int,
    m: int,
) -> np.ndarray:
    """
    Build the complete LMI matrix: S + Ñ₁ + Ñ₂.

    The main LMI condition from Theorem 1 is:
        S(P, L, nu) + Ñ₁(data, beta) + Ñ₂(C, T̃) ≺ 0

    Args:
        P: (n, n) Funnel shape matrix
        L: (m, n) Feedback parameter
        nu: S-procedure multiplier
        alpha: Contraction rate
        H, H_plus, Xi: Data matrices
        beta: Disturbance bound
        C: Jacobian variation rate
        T_tilde: Maximum index difference
        n, m: Dimensions

    Returns:
        LMI: (4n+2m, 4n+2m) matrix that should be negative semidefinite
    """
    S = build_S_matrix(P, L, nu, alpha, n, m)
    N1 = build_N1_tilde(H, H_plus, Xi, beta, n, m)
    N2 = build_N2_tilde(C, T_tilde, n, m)

    return S + N1 + N2


def verify_lmi_dimensions(n: int = 4, m: int = 2) -> dict:
    """
    Verify LMI matrix dimensions match expected values.

    Args:
        n: State dimension (default 4)
        m: Input dimension (default 2)

    Returns:
        dict with dimension information
    """
    dim = get_total_dimension(n, m)
    idx = get_block_indices(n, m)
    block_sizes = [idx[i+1] - idx[i] for i in range(6)]

    return {
        "total_dimension": dim,
        "expected_dimension": 20,
        "block_indices": idx,
        "block_sizes": block_sizes,
        "expected_block_sizes": [n, n, m, n, m, n],
        "matches": dim == 20 and block_sizes == [4, 4, 2, 4, 2, 4],
    }


def build_constraint_matrices_cvxpy(
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    beta: float,
    C: float,
    T_tilde: int,
    n: int,
    m: int,
):
    """
    Build constant matrices for CVXPY formulation.

    In CVXPY, S contains variables (P, L, nu) while Ñ₁, Ñ₂ are constants.
    This function returns the constant parts.

    Args:
        H, H_plus, Xi: Data matrices
        beta: Disturbance bound
        C: Jacobian variation rate
        T_tilde: Maximum index difference
        n, m: Dimensions

    Returns:
        N1_tilde: Constant data constraint matrix
        N2_tilde: Constant Jacobian variation matrix
    """
    N1_tilde = build_N1_tilde(H, H_plus, Xi, beta, n, m)
    N2_tilde = build_N2_tilde(C, T_tilde, n, m)

    return N1_tilde, N2_tilde
