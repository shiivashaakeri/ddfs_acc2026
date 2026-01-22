"""
LMI Matrix Construction.

Build S, N1_tilde, N2_tilde matrices for the main LMI (Theorem 1).

Block structure for S (20x20):
    Blocks: [n, n, m, n, m, n] = [4, 4, 2, 4, 2, 4]
    Row/col indices: [0, 4, 8, 10, 14, 16, 20]

S(P_i, L_i, nu) structure:
    [alpha*P - nu*I,  0,     0,     0,     0,     0    ]
    [0,             -P,   -L^T,   -P,   -L^T,    0    ]
    [0,             -L,     0,    -L,     0,     L    ]
    [0,             -P,   -L^T,   -P,   -L^T,    0    ]
    [0,             -L,     0,    -L,     0,     L    ]
    [0,              0,    L^T,    0,    L^T,    P    ]
"""

# TODO: Implement LMI matrix builders
# def build_S_matrix(P, L, nu, alpha, n, m) -> np.ndarray:
#     """
#     Build 6-block S matrix.
#     Total dimension: 4n + 2m = 20 for our case
#     """

# def build_N1_tilde(H, H_plus, Xi, beta, n, m) -> np.ndarray:
#     """
#     Build N1_tilde matrix (data constraint).
#     N1 = S_W * M_W * S_W^T padded to 6-block size
#     """

# def build_N2_tilde(C, T_tilde, n, m) -> np.ndarray:
#     """
#     Build N2_tilde matrix (Jacobian variation constraint).
#     N2 = S_Delta^T * M_Delta * S_Delta padded to 6-block size
#     """
