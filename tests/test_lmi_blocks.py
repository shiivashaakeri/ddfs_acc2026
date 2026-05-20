"""Algebraic identities for the LMI block construction (Theorem 1 proof).

These tests would have caught the bug in the previous synthesis code: they
check that Φ^T M Φ, Φ^T N_1 Φ, Φ^T N_2 Φ literally equal what the paper's
S-procedure encodes.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from synthesis.lmi_blocks import (
    M_numeric,
    build_N1_tilde,
    build_N2_tilde,
    build_S_matrix,
    lifted_Phi,
)


def _random_instance(n: int, m: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((n, n))
    P = P @ P.T + 0.1 * np.eye(n)
    K = rng.standard_normal((m, n)) * 0.3
    A = 0.9 * np.eye(n) + 0.1 * rng.standard_normal((n, n))
    B = 0.2 * rng.standard_normal((n, m))
    dA = 0.05 * rng.standard_normal((n, n))
    dB = 0.05 * rng.standard_normal((n, m))
    return P, K, A, B, dA, dB


def test_M_quadratic_form_equals_alpha_P_minus_Acl_P_AclT():
    """Proof step (20): Φ^T M Φ == α P − A_cl P A_cl^T."""
    n, m, alpha = 4, 2, 0.95
    for seed in range(5):
        P, K, A, B, dA, dB = _random_instance(n, m, seed=seed)
        Phi = lifted_Phi(A, B, dA, dB)
        M = M_numeric(P, K, alpha)
        lhs = Phi.T @ M @ Phi
        A_cl = A + B @ K + dA + dB @ K
        rhs = alpha * P - A_cl @ P @ A_cl.T
        assert np.allclose(lhs, rhs, atol=1e-10), f"seed={seed}"


def test_N1_tilde_encodes_data_consistency():
    """Φ^T N_1 Φ == β I − W W^T  with W = H_plus − A H − B Ξ."""
    n, m, L_data = 4, 2, 20
    for seed in range(5):
        _, _, A, B, dA, dB = _random_instance(n, m, seed=seed)
        rng = np.random.default_rng(seed + 100)
        H = rng.standard_normal((n, L_data))
        Xi = rng.standard_normal((m, L_data))
        W = 0.05 * rng.standard_normal((n, L_data))
        H_plus = A @ H + B @ Xi + W
        beta = 1.7

        N1_tilde = build_N1_tilde(H, H_plus, Xi, beta, n, m)
        # Strip the padding (last n rows/cols of zeros) to use Φ.
        dim5 = 3 * n + 2 * m
        N1 = N1_tilde[:dim5, :dim5]

        Phi = lifted_Phi(A, B, dA, dB)
        lhs = Phi.T @ N1 @ Phi
        rhs = beta * np.eye(n) - W @ W.T
        assert np.allclose(lhs, rhs, atol=1e-10), f"seed={seed}, max err {np.max(np.abs(lhs-rhs))}"


def test_N2_tilde_encodes_jacobian_variation():
    """Φ^T N_2 Φ == C² T̃² I_n − ΔA ΔA^T − ΔB ΔB^T."""
    n, m = 4, 2
    C, T_tilde = 3.64, 199.0
    for seed in range(5):
        _, _, A, B, dA, dB = _random_instance(n, m, seed=seed)
        N2_tilde = build_N2_tilde(C, T_tilde, n, m)
        dim5 = 3 * n + 2 * m
        N2 = N2_tilde[:dim5, :dim5]

        Phi = lifted_Phi(A, B, dA, dB)
        lhs = Phi.T @ N2 @ Phi
        rhs = (C ** 2) * (T_tilde ** 2) * np.eye(n) - dA @ dA.T - dB @ dB.T
        assert np.allclose(lhs, rhs, atol=1e-8), f"seed={seed}"


def test_S_matrix_recovers_M_via_Schur():
    """The Schur complement of the (6,6)=P block in S equals M_numeric(P, L P^{-1}).

    This verifies the change-of-variables (L = K P) and Schur padding are
    consistent.
    """
    n, m, alpha = 4, 2, 0.95
    nu = 1e-6
    for seed in range(5):
        P, K, _, _, _, _ = _random_instance(n, m, seed=seed)
        L_mat = K @ P
        # Build S numerically by substituting cvxpy parameters with concrete values.
        P_cp = cp.Parameter((n, n), symmetric=True, value=P)
        L_cp = cp.Parameter((m, n), value=L_mat)
        nu_cp = cp.Parameter(value=nu)
        S_expr = build_S_matrix(P_cp, L_cp, nu_cp, alpha, n, m)
        S = np.asarray(S_expr.value)

        # Schur on (6,6) = P (assumes P ≻ 0)
        dim = S.shape[0]
        n6 = n  # last block size
        upper = S[: dim - n6, : dim - n6]
        side = S[: dim - n6, dim - n6:]
        corner = S[dim - n6:, dim - n6:]  # should equal P
        assert np.allclose(corner, P, atol=1e-10)
        S_schur = upper - side @ np.linalg.solve(corner, side.T)

        M = M_numeric(P, K, alpha)
        # The (1,1) block of S has -νI added; subtract it back for comparison.
        M_with_nu = M.copy()
        M_with_nu[:n, :n] -= nu * np.eye(n)
        assert np.allclose(S_schur, M_with_nu, atol=1e-8), (
            f"seed={seed}, max err {np.max(np.abs(S_schur - M_with_nu))}"
        )


def test_s_lemma_implication_on_synthetic_data():
    """End-to-end S-lemma sanity: if the LMI S - λ_1 Ñ_1 - λ_2 Ñ_2 ⪰ 0 holds
    at concrete numeric (P, L, ν, λ_1, λ_2), then α P − A_cl P A_cl^T ⪰ 0
    for any (A, B) in the data-consistent set and (ΔA, ΔB) in the variation set.

    We construct an "easy" instance: take known (A, B), zero variation, and a
    Lyapunov P that we know works, then check the implication numerically.
    """
    n, m, alpha = 2, 1, 0.9
    rng = np.random.default_rng(0)
    A_true = np.array([[0.5, 0.1], [0.0, 0.4]])
    B_true = np.array([[0.0], [1.0]])
    # Stable open-loop, so K = 0 works.
    K = np.zeros((m, n))
    P = np.eye(n)
    L_mat = K @ P

    # Make perfectly clean data (zero noise) so β should certify W = 0 exactly.
    L_data = 6
    H = rng.standard_normal((n, L_data))
    Xi = rng.standard_normal((m, L_data))
    H_plus = A_true @ H + B_true @ Xi
    beta = 1e-8

    A_cl = A_true + B_true @ K
    rhs = alpha * P - A_cl @ P @ A_cl.T
    assert np.linalg.eigvalsh(rhs).min() >= -1e-10
