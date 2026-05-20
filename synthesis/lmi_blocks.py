"""LMI block construction for Theorem 1 (Eq. 19).

This module is a literal translation of the matrix S-lemma reduction in
Section V.B. Every block matches the paper symbol-for-symbol, and the
correctness is checked by an algebraic identity test (test_lmi_blocks.py):

    Φ^T M Φ  ==  α P − A_cl P A_cl^T     (the original QMI from the proof)
    Φ^T N_1 Φ == β I − W W^T              (data-consistency premise)
    Φ^T N_2 Φ == C² T̃² I − ΔA ΔA^T − ΔB ΔB^T   (Jacobian variation premise)

where Φ = [I; A^T; B^T; ΔA^T; ΔB^T] is the lifted stack and
A_cl = A + B K + ΔA + ΔB K.

The "S(P, L, ν)" matrix is the Schur-padded reformulation of M that becomes
affine in (P, L, ν) under the change of variables L = K P. Padding adds one
extra n-sized block-row/column (the 6th block). Ñ_1, Ñ_2 are N_1, N_2 with a
matching zero-padded n-block.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


# Block sizes for the 6-block S matrix: (n, n, m, n, m, n).
def _block_offsets(n: int, m: int) -> list[int]:
    sizes = [n, n, m, n, m, n]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    return offsets  # length 7; offsets[i+1] - offsets[i] = sizes[i]


def _block_offsets_5(n: int, m: int) -> list[int]:
    """5-block original Φ-space offsets (n, n, m, n, m)."""
    sizes = [n, n, m, n, m]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    return offsets


def build_S_matrix(P, L, nu, alpha: float, n: int, m: int):
    """Symbolic S(P, L, ν) of dimension (4n + 2m) × (4n + 2m).

    Layout (block indices 1..6, sizes n, n, m, n, m, n):
        S[1,1] =  αP − νI
        S[2,2] = S[2,4] = S[4,2] = S[4,4] = −P
        S[2,3] = S[2,5] = S[4,3] = S[4,5] = −L^T
        S[3,2] = S[3,4] = S[5,2] = S[5,4] = −L
        S[3,3] = S[3,5] = S[5,3] = S[5,5] = 0          (eliminated via Schur on block 6)
        S[3,6] = S[5,6] = L
        S[6,3] = S[6,5] = L^T
        S[6,6] = P
    """
    o = _block_offsets(n, m)
    dim = o[-1]

    # Build a list of (slice_row, slice_col, expr) entries.
    blocks: list[tuple[slice, slice, object]] = []

    def add(i: int, j: int, expr):
        blocks.append((slice(o[i - 1], o[i]), slice(o[j - 1], o[j]), expr))

    I_n = np.eye(n)
    add(1, 1, alpha * P - nu * I_n)

    minus_P = -P
    minus_LT = -cp.transpose(L)
    minus_L = -L

    for (i, j) in [(2, 2), (2, 4), (4, 2), (4, 4)]:
        add(i, j, minus_P)
    for (i, j) in [(2, 3), (2, 5), (4, 3), (4, 5)]:
        add(i, j, minus_LT)
    for (i, j) in [(3, 2), (3, 4), (5, 2), (5, 4)]:
        add(i, j, minus_L)
    # (3,3), (3,5), (5,3), (5,5), (1,2..5,2..5), (1,6), etc. remain zero
    add(3, 6, L)
    add(5, 6, L)
    add(6, 3, cp.transpose(L))
    add(6, 5, cp.transpose(L))
    add(6, 6, P)

    # Assemble row-by-row using cp.bmat
    n_blocks = 6
    row_dims = [n, n, m, n, m, n]
    col_dims = row_dims
    grid: list[list[object]] = [
        [None for _ in range(n_blocks)] for _ in range(n_blocks)
    ]
    # Fill zeros first
    for i in range(n_blocks):
        for j in range(n_blocks):
            grid[i][j] = np.zeros((row_dims[i], col_dims[j]))
    # Overlay non-zero blocks (entries are 1-indexed in `add`)
    for (rs, cs, expr) in blocks:
        # Find which block (i,j) this slice corresponds to
        bi = o.index(rs.start)
        bj = o.index(cs.start)
        grid[bi][bj] = expr
    return cp.bmat(grid)


def build_N1_tilde(H: np.ndarray, H_plus: np.ndarray, Xi: np.ndarray, beta: float, n: int, m: int) -> np.ndarray:
    """Ñ_1 of dimension (4n + 2m) × (4n + 2m), data-consistency premise.

    N_1 = S_W diag(β I_n, -I_L) S_W^T  in the original 5-block space (3n+2m),
    where
        S_W = [[I_n,  H_plus],
               [0,   -H    ],
               [0,   -Ξ    ],
               [0,    0    ],
               [0,    0    ]]   (block-rows of sizes n, n, m, n, m).

    Ñ_1 is N_1 padded with a zero 6th n-block to match S's dimension.
    """
    L = H.shape[1]
    o5 = _block_offsets_5(n, m)  # length 6; total dim 3n+2m
    dim5 = o5[-1]

    S_W = np.zeros((dim5, n + L))
    # Block-row 1 (size n): [I_n, H_plus]
    S_W[o5[0]:o5[1], :n] = np.eye(n)
    S_W[o5[0]:o5[1], n:] = H_plus
    # Block-row 2 (size n): [0, -H]
    S_W[o5[1]:o5[2], n:] = -H
    # Block-row 3 (size m): [0, -Ξ]
    S_W[o5[2]:o5[3], n:] = -Xi
    # Block-rows 4 (size n) and 5 (size m) are all zero.

    M_W = np.zeros((n + L, n + L))
    M_W[:n, :n] = beta * np.eye(n)
    M_W[n:, n:] = -np.eye(L)

    N1 = S_W @ M_W @ S_W.T  # (3n+2m, 3n+2m)

    # Pad to S's dimension (4n+2m) by adding a zero 6th block-row/column of size n.
    o6 = _block_offsets(n, m)
    dim6 = o6[-1]
    N1_tilde = np.zeros((dim6, dim6))
    N1_tilde[:dim5, :dim5] = N1
    return N1_tilde


def build_N2_tilde(C: float, T_tilde: float, n: int, m: int) -> np.ndarray:
    """Ñ_2: Jacobian variation premise [ΔA ΔB][ΔA;ΔB]^T ⪯ C² T̃² I_n.

    N_2 = S_Δ diag(C²T̃² I_n, -I_n, -I_m) S_Δ^T   in the 5-block space, where
        S_Δ = [[I_n, 0, 0],
               [ 0,  0, 0],
               [ 0,  0, 0],
               [ 0, I_n, 0],
               [ 0,  0, I_m]].

    Ñ_2 pads with a zero 6th n-block.
    """
    o5 = _block_offsets_5(n, m)
    dim5 = o5[-1]

    S_D = np.zeros((dim5, n + n + m))
    # Block-row 1: I_n in the first n cols
    S_D[o5[0]:o5[1], :n] = np.eye(n)
    # Block-rows 2, 3 are zero
    # Block-row 4 (size n): I_n in the middle n cols
    S_D[o5[3]:o5[4], n:2 * n] = np.eye(n)
    # Block-row 5 (size m): I_m in the last m cols
    S_D[o5[4]:o5[5], 2 * n:] = np.eye(m)

    M_D = np.zeros((n + n + m, n + n + m))
    M_D[:n, :n] = (C ** 2) * (T_tilde ** 2) * np.eye(n)
    M_D[n:2 * n, n:2 * n] = -np.eye(n)
    M_D[2 * n:, 2 * n:] = -np.eye(m)

    N2 = S_D @ M_D @ S_D.T  # (3n+2m, 3n+2m)

    o6 = _block_offsets(n, m)
    dim6 = o6[-1]
    N2_tilde = np.zeros((dim6, dim6))
    N2_tilde[:dim5, :dim5] = N2
    return N2_tilde


# ---------------------------------------------------------------------------
# Reference (numeric) constructions for testing the proof's QMI identities.
# These rebuild M(P, K), N_1, N_2 in the 5-block Φ-space using numerical
# matrices (not cvxpy) so we can check the algebraic identities exactly.
# ---------------------------------------------------------------------------


def lifted_Phi(A: np.ndarray, B: np.ndarray, dA: np.ndarray, dB: np.ndarray) -> np.ndarray:
    """Φ = [I_n; A^T; B^T; ΔA^T; ΔB^T] as a (3n + 2m) × n matrix."""
    n, m = A.shape[0], B.shape[1]
    return np.vstack([np.eye(n), A.T, B.T, dA.T, dB.T])


def M_numeric(P: np.ndarray, K: np.ndarray, alpha: float) -> np.ndarray:
    """Original M in the proof: 5-block matrix in (P, K) numeric form."""
    n, m = P.shape[0], K.shape[0]
    o5 = _block_offsets_5(n, m)
    dim5 = o5[-1]
    M = np.zeros((dim5, dim5))
    M[o5[0]:o5[1], o5[0]:o5[1]] = alpha * P

    minus_P = -P
    minus_PKT = -P @ K.T
    minus_KP = -K @ P
    minus_KPKT = -K @ P @ K.T

    for (i, j) in [(2, 2), (2, 4), (4, 2), (4, 4)]:
        M[o5[i - 1]:o5[i], o5[j - 1]:o5[j]] = minus_P
    for (i, j) in [(2, 3), (2, 5), (4, 3), (4, 5)]:
        M[o5[i - 1]:o5[i], o5[j - 1]:o5[j]] = minus_PKT
    for (i, j) in [(3, 2), (3, 4), (5, 2), (5, 4)]:
        M[o5[i - 1]:o5[i], o5[j - 1]:o5[j]] = minus_KP
    for (i, j) in [(3, 3), (3, 5), (5, 3), (5, 5)]:
        M[o5[i - 1]:o5[i], o5[j - 1]:o5[j]] = minus_KPKT
    return M
