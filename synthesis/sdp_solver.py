"""
Main SDP Solver (Eq. 22).

Solve the convex semidefinite program for funnel synthesis:
    maximize    log det(P_i)
    subject to  P_i ≻ 0
                λ₁ ≥ 0, λ₂ ≥ 0, ν > 0
                S(P_i, L_i, ν) - λ₁·Ñ₁ - λ₂·Ñ₂ ≽ 0
                P_i ≽ P_min,i
                [R_max,i, L_i; L_i^T, P_i] ≽ 0
                P_i ≼ μ·P_prev  (if not first segment)

The objective maximizes the funnel volume (log det P).
The constraints ensure:
- Funnel contraction (main LMI)
- State constraint satisfaction (P ≥ P_min)
- Input constraint satisfaction (Schur complement)
- Funnel continuity across segments (P ≤ μ·P_prev)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np

from synthesis.lmi_matrices import get_block_indices, get_total_dimension


def estimate_AB_from_data(
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    n: int,
    m: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate linearized dynamics (A_hat, B_hat) from data matrices.

    From the relation: H_plus ≈ A @ H + B @ Xi + D
    We solve: [A_hat, B_hat] = H_plus @ [H; Xi]^†

    Args:
        H: (n, L) State deviation matrix
        H_plus: (n, L) Next-state deviation matrix
        Xi: (m, L) Input deviation matrix
        n: State dimension
        m: Input dimension

    Returns:
        A_hat: (n, n) Estimated state matrix, or None if estimation fails
        B_hat: (n, m) Estimated input matrix, or None if estimation fails
    """
    try:
        # Stack [H; Xi] -> (n+m, L)
        Z = np.vstack([H, Xi])

        # Solve least squares: H_plus = [A_hat, B_hat] @ Z
        # => [A_hat, B_hat]^T = Z^† @ H_plus^T
        # Using pseudoinverse
        Z_pinv = np.linalg.pinv(Z.T)  # (n+m, L)^† -> (L, n+m)
        AB_hat = (Z_pinv @ H_plus.T).T  # (n, n+m)

        A_hat = AB_hat[:, :n]
        B_hat = AB_hat[:, n:n+m]

        # Verify dimensions
        if A_hat.shape != (n, n) or B_hat.shape != (n, m):
            return None, None

        return A_hat, B_hat

    except np.linalg.LinAlgError:
        return None, None


def compute_stabilizing_L_from_data(
    A_hat: np.ndarray,
    B_hat: np.ndarray,
    P_target: np.ndarray,
    n: int,
    m: int,
) -> Optional[np.ndarray]:
    """
    Compute a stabilizing L = K @ P from estimated dynamics.

    Uses discrete-time LQR on (A_hat, B_hat) to get stabilizing K,
    then computes L = K @ P_target.

    Args:
        A_hat: (n, n) Estimated state matrix
        B_hat: (n, m) Estimated input matrix
        P_target: (n, n) Target P matrix (e.g., P_min)
        n: State dimension
        m: Input dimension

    Returns:
        L_target: (m, n) Stabilizing feedback parameter, or None if fails
    """
    try:
        from scipy.linalg import solve_discrete_are

        # LQR weights
        Q = np.eye(n) * 10.0
        R = np.eye(m) * 0.1

        # Solve discrete-time algebraic Riccati equation
        P_lqr = solve_discrete_are(A_hat, B_hat, Q, R)

        # LQR gain: K = -(R + B^T P B)^{-1} B^T P A
        K_lqr = -np.linalg.solve(R + B_hat.T @ P_lqr @ B_hat, B_hat.T @ P_lqr @ A_hat)

        # For our convention u = u_nom + K @ eta, we negate
        K_stab = -K_lqr

        # Compute L = K @ P
        L_target = K_stab @ P_target

        return L_target

    except Exception:
        return None


@dataclass
class SDPResult:
    """Container for SDP solver results."""

    P: np.ndarray  # (n, n) optimal funnel shape
    K: np.ndarray  # (m, n) feedback gain
    L: np.ndarray  # (m, n) L = K @ P
    nu: float  # S-procedure multiplier
    lambda1: float  # Data constraint multiplier
    lambda2: float  # Jacobian variation multiplier
    success: bool  # Solver converged
    status: str  # Solver status string
    objective: float  # Optimal objective value


def build_S_matrix_cvxpy(
    P: cp.Variable,
    L: cp.Variable,
    nu: cp.Variable,
    alpha: float,
    n: int,
    m: int,
) -> cp.Expression:
    """
    Build the S matrix as a CVXPY expression.

    Block structure (symmetric):
        [alpha*P - nu*I,    0,       0,       0,       0,      0   ]
        [     0,          -P,     -L^T,     -P,     -L^T,      0   ]
        [     0,          -L,       0,      -L,       0,       L   ]
        [     0,          -P,     -L^T,     -P,     -L^T,      0   ]
        [     0,          -L,       0,      -L,       0,       L   ]
        [     0,           0,      L^T,      0,      L^T,      P   ]

    Args:
        P: (n, n) CVXPY Variable (symmetric)
        L: (m, n) CVXPY Variable
        nu: scalar CVXPY Variable
        alpha: contraction rate
        n, m: dimensions

    Returns:
        S: CVXPY block matrix expression
    """
    # Identity matrices
    In = np.eye(n)
    Im = np.eye(m)

    # Zero matrices of appropriate sizes
    Z_nn = np.zeros((n, n))
    Z_nm = np.zeros((n, m))
    Z_mn = np.zeros((m, n))
    Z_mm = np.zeros((m, m))

    # Build S using cp.bmat (block matrix)
    # Row 0: [alpha*P - nu*I, 0, 0, 0, 0, 0]
    # Row 1: [0, -P, -L^T, -P, -L^T, 0]
    # Row 2: [0, -L, 0, -L, 0, L]
    # Row 3: [0, -P, -L^T, -P, -L^T, 0]
    # Row 4: [0, -L, 0, -L, 0, L]
    # Row 5: [0, 0, L^T, 0, L^T, P]

    S = cp.bmat([
        [alpha * P - nu * In, Z_nn, Z_nm, Z_nn, Z_nm, Z_nn],
        [Z_nn, -P, -L.T, -P, -L.T, Z_nn],
        [Z_mn, -L, Z_mm, -L, Z_mm, L],
        [Z_nn, -P, -L.T, -P, -L.T, Z_nn],
        [Z_mn, -L, Z_mm, -L, Z_mm, L],
        [Z_nn, Z_nn, L.T, Z_nn, L.T, P],
    ])

    return S


def solve_funnel_sdp(
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    beta: float,
    P_min: np.ndarray,
    R_max: np.ndarray,
    alpha: float,
    C: float,
    T_tilde: int,
    n: int,
    m: int,
    P_prev: Optional[np.ndarray] = None,
    mu: float = 1.02,
    solver: str = "SCS",
    verbose: bool = False,
    L_reg_weight: float = 0.1,
    use_data_stability: bool = True,
) -> SDPResult:
    """
    Solve the main SDP for segment i (Eq. 22).

    maximize    log det(P)
    subject to  P ≻ 0
                λ₁ ≥ 0, λ₂ ≥ 0, ν > 0
                S(P, L, ν) - λ₁·Ñ₁ - λ₂·Ñ₂ ≽ 0
                P ≽ P_min
                [R_max, L; L^T, P] ≽ 0
                P ≼ μ·P_prev  (if not first segment)

    Args:
        H: (n, L) State deviation matrix from data
        H_plus: (n, L) Next-state deviation matrix
        Xi: (m, L) Input deviation matrix
        beta: Disturbance energy bound
        P_min: (n, n) Minimum P from state constraints
        R_max: (m, m) Maximum R from input constraints
        alpha: Contraction rate (e.g., 0.98)
        C: Jacobian variation rate
        T_tilde: Maximum index difference
        n: State dimension
        m: Input dimension
        P_prev: (n, n) Previous segment P (for continuity)
        mu: Growth bound (e.g., 1.02)
        solver: CVXPY solver to use
        verbose: Print solver output
        L_reg_weight: Weight for L regularization in objective (default 0.1)
        use_data_stability: If True, use estimated A,B from data for L regularization (default True)

    Returns:
        SDPResult with P, K, L, multipliers, and status
    """
    # Build constant matrices Ñ₁ and Ñ₂
    from synthesis.lmi_matrices import build_N1_tilde, build_N2_tilde

    N1_tilde = build_N1_tilde(H, H_plus, Xi, beta, n, m)
    N2_tilde = build_N2_tilde(C, T_tilde, n, m)

    # Decision variables
    P = cp.Variable((n, n), symmetric=True)
    L = cp.Variable((m, n))
    nu = cp.Variable(nonneg=True)
    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)

    # Build S matrix expression
    S = build_S_matrix_cvxpy(P, L, nu, alpha, n, m)

    # Constraints
    constraints = []

    # 1. P positive definite (with small margin for numerical stability)
    eps_pd = 1e-6
    constraints.append(P >> eps_pd * np.eye(n))

    # 2. nu strictly positive
    constraints.append(nu >= 1e-6)

    # 3. Main LMI: S - λ₁·Ñ₁ - λ₂·Ñ₂ ≽ 0
    # Note: We subtract the constraint matrices (scaled by multipliers)
    main_lmi = S - lambda1 * N1_tilde - lambda2 * N2_tilde
    constraints.append(main_lmi >> 0)

    # 4. State constraint: P ≽ P_min
    constraints.append(P >> P_min)

    # 5. Input constraint via Schur complement: [R_max, L; L^T, P] ≽ 0
    # This ensures L @ P^{-1} @ L^T ≤ R_max (bounded control effort)
    input_schur = cp.bmat([
        [R_max, L],
        [L.T, P]
    ])
    constraints.append(input_schur >> 0)

    # 6. Continuity constraint: P ≼ μ·P_prev (if not first segment)
    if P_prev is not None:
        constraints.append(P << mu * P_prev)

    # ========== REGULARIZATION TO PREVENT L ≈ 0 ==========

    # 7. Data-driven stability regularization (optional)
    # Estimate A_hat, B_hat from data and compute stabilizing L_target
    L_target = None
    if use_data_stability and H.shape[1] >= n + m:
        A_hat, B_hat = estimate_AB_from_data(H, H_plus, Xi, n, m)
        if A_hat is not None and B_hat is not None:
            L_target = compute_stabilizing_L_from_data(A_hat, B_hat, P_min, n, m)

    # Objective: maximize log det(P) + regularization on L
    # Option A: Encourage L to be large (prevents trivial L=0)
    #   maximize log det(P) + L_reg_weight * ||L||_F^2
    # Option B: Encourage L to match stabilizing target (if available)
    #   maximize log det(P) - L_reg_weight * ||L - L_target||_F^2
    #
    # Note: cp.sum_squares(L) = ||L||_F^2 = tr(L @ L.T) and is DCP-compliant

    if L_target is not None and np.linalg.norm(L_target) > 1e-6:
        # Encourage L to match the LQR-derived stabilizing target
        # Minimize ||L - L_target||^2, which means maximize -||L - L_target||^2
        objective = cp.Maximize(
            cp.log_det(P) - L_reg_weight * cp.sum_squares(L - L_target)
        )
    else:
        # Encourage larger L magnitude to prevent trivial solutions
        objective = cp.Maximize(cp.log_det(P) + L_reg_weight * cp.sum_squares(L))

    # Solve
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=getattr(cp, solver), verbose=verbose)
    except cp.error.SolverError as e:
        # Try alternative solver
        try:
            problem.solve(solver=cp.CVXOPT, verbose=verbose)
        except Exception:
            return SDPResult(
                P=np.eye(n),
                K=np.zeros((m, n)),
                L=np.zeros((m, n)),
                nu=0.0,
                lambda1=0.0,
                lambda2=0.0,
                success=False,
                status=f"Solver error: {e}",
                objective=-np.inf,
            )

    # Check solution status
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        P_opt = P.value
        L_opt = L.value

        # Compute feedback gain K = L @ P^{-1}
        try:
            K_opt = L_opt @ np.linalg.inv(P_opt)
        except np.linalg.LinAlgError:
            K_opt = np.zeros((m, n))

        return SDPResult(
            P=P_opt,
            K=K_opt,
            L=L_opt,
            nu=float(nu.value),
            lambda1=float(lambda1.value),
            lambda2=float(lambda2.value),
            success=True,
            status=problem.status,
            objective=problem.value,
        )
    else:
        return SDPResult(
            P=np.eye(n),
            K=np.zeros((m, n)),
            L=np.zeros((m, n)),
            nu=0.0,
            lambda1=0.0,
            lambda2=0.0,
            success=False,
            status=problem.status,
            objective=-np.inf,
        )


def solve_initial_segment_sdp(
    P_min: np.ndarray,
    R_max: np.ndarray,
    n: int,
    m: int,
    solver: str = "SCS",
    verbose: bool = False,
) -> SDPResult:
    """
    Solve simplified SDP for the initial segment (no data yet).

    For the first segment, we don't have data matrices, so we solve:
        maximize    log det(P)
        subject to  P ≻ 0
                    P ≽ P_min
                    [R_max, L; L^T, P] ≽ 0

    This gives an initial P₀ and L₀ to start the controller.

    Args:
        P_min: (n, n) Minimum P from state constraints
        R_max: (m, m) Maximum R from input constraints
        n, m: dimensions
        solver: CVXPY solver
        verbose: Print output

    Returns:
        SDPResult with initial P, K, L
    """
    # Decision variables
    P = cp.Variable((n, n), symmetric=True)
    L = cp.Variable((m, n))

    # Constraints
    constraints = []

    # P positive definite
    eps_pd = 1e-6
    constraints.append(P >> eps_pd * np.eye(n))

    # State constraint
    constraints.append(P >> P_min)

    # Input constraint (Schur complement)
    input_schur = cp.bmat([
        [R_max, L],
        [L.T, P]
    ])
    constraints.append(input_schur >> 0)

    # Objective
    objective = cp.Maximize(cp.log_det(P))

    # Solve
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=getattr(cp, solver), verbose=verbose)
    except cp.error.SolverError:
        try:
            problem.solve(solver=cp.CVXOPT, verbose=verbose)
        except Exception as e:
            return SDPResult(
                P=np.eye(n),
                K=np.zeros((m, n)),
                L=np.zeros((m, n)),
                nu=0.0,
                lambda1=0.0,
                lambda2=0.0,
                success=False,
                status=str(e),
                objective=-np.inf,
            )

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        P_opt = P.value
        L_opt = L.value

        try:
            K_opt = L_opt @ np.linalg.inv(P_opt)
        except np.linalg.LinAlgError:
            K_opt = np.zeros((m, n))

        return SDPResult(
            P=P_opt,
            K=K_opt,
            L=L_opt,
            nu=0.0,
            lambda1=0.0,
            lambda2=0.0,
            success=True,
            status=problem.status,
            objective=problem.value,
        )
    else:
        return SDPResult(
            P=np.eye(n),
            K=np.zeros((m, n)),
            L=np.zeros((m, n)),
            nu=0.0,
            lambda1=0.0,
            lambda2=0.0,
            success=False,
            status=problem.status,
            objective=-np.inf,
        )


def verify_funnel_constraints(
    P: np.ndarray,
    L: np.ndarray,
    P_min: np.ndarray,
    R_max: np.ndarray,
    P_prev: Optional[np.ndarray] = None,
    mu: float = 1.02,
) -> dict:
    """
    Verify that funnel matrices satisfy all constraints.

    Args:
        P: (n, n) Funnel shape matrix
        L: (m, n) Feedback parameter
        P_min: (n, n) Minimum P bound
        R_max: (m, m) Maximum R bound
        P_prev: Previous P (optional)
        mu: Growth bound

    Returns:
        dict with constraint satisfaction info
    """
    n, m = P.shape[0], L.shape[0]

    # Check P positive definite
    P_eigvals = np.linalg.eigvalsh(P)
    P_pd = P_eigvals.min() > 0

    # Check P >= P_min
    diff_Pmin = P - P_min
    diff_Pmin_eigvals = np.linalg.eigvalsh(diff_Pmin)
    P_geq_Pmin = diff_Pmin_eigvals.min() >= -1e-8

    # Check Schur complement
    schur = np.block([
        [R_max, L],
        [L.T, P]
    ])
    schur_eigvals = np.linalg.eigvalsh(schur)
    schur_psd = schur_eigvals.min() >= -1e-8

    # Check continuity
    if P_prev is not None:
        diff_Pprev = mu * P_prev - P
        diff_Pprev_eigvals = np.linalg.eigvalsh(diff_Pprev)
        P_leq_muPprev = diff_Pprev_eigvals.min() >= -1e-8
    else:
        P_leq_muPprev = True

    return {
        "P_positive_definite": P_pd,
        "P_min_eigenvalue": P_eigvals.min(),
        "P_geq_P_min": P_geq_Pmin,
        "schur_psd": schur_psd,
        "schur_min_eigenvalue": schur_eigvals.min(),
        "continuity_satisfied": P_leq_muPprev,
        "all_satisfied": P_pd and P_geq_Pmin and schur_psd and P_leq_muPprev,
    }
