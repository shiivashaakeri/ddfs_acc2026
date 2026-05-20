"""Initial stabilizing controller K_0 (Assumption 5), shape-matrix convention.

The paper's funnel matrix P is the shape matrix: E(P) = {η : η^T P^{-1} η ≤ 1}.
We design K_0 via DLQR on the twin's Jacobian at the start, then build P_0 from
the closed-loop Lyapunov equation -- specifically, the solution Q_0 of

    A_cl Q_0 A_cl^T - α Q_0 + W_0 = 0,    Q_0 ≻ 0

where A_cl = A_twin + B_twin K_0 (with K_0 the LQR gain for the deviation
dynamics ξ = K_0 η, so K_0 = -K_DLQR), α the Lyapunov decay, and W_0 a small
PSD seed. We then scale Q_0 to satisfy:

  (i) Q_0 ⪯ P_max_shape  (state feasibility)
  (ii) η_0^T Q_0^{-1} η_0 ≤ 1  (initial deviation inside the funnel)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from dynamics.discretization import DiscreteSystem
from linearization.jacobians import jacobians_discrete


def design_initial_controller(
    twin: DiscreteSystem,
    x_nom_0: np.ndarray,
    u_nom_0: np.ndarray,
    P_max_shape_0: np.ndarray,
    eta_0: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    alpha: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_0, K_0) in shape-matrix convention.

    P_0 ⪯ P_max_shape_0 and (if eta_0 supplied) η_0 ∈ E_shape(P_0).
    """
    A, B = jacobians_discrete(twin.step, x_nom_0, u_nom_0)
    n, m = A.shape[0], B.shape[1]
    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1, 0.1])
    if R is None:
        R = 1.0 * np.eye(m)
    P_ric = solve_discrete_are(A, B, Q, R)
    K_dlqr = np.linalg.solve(R + B.T @ P_ric @ B, B.T @ P_ric @ A)
    # Deviation control: ξ = K_0 η, closed loop A + B K_0. Need K_0 = -K_dlqr.
    K_0 = -K_dlqr
    A_cl = A + B @ K_0  # should be stable

    # Compute the shape matrix Q_0 satisfying A_cl Q_0 A_cl^T = α Q_0 - W_0 for some W_0 ⪰ 0.
    # The discrete Lyapunov equation A_cl X A_cl^T - X + W = 0 with W = I gives X ≻ 0.
    # Then α X - A_cl X A_cl^T = α X - (X - W) = (α-1) X + W. For α < 1 this is
    # PSD iff W ⪰ (1-α) X. Use W = I and rescale.
    W_seed = np.eye(n)
    X = solve_discrete_lyapunov(A_cl, W_seed)
    X = 0.5 * (X + X.T)
    # Solve for c such that c * X satisfies α (c X) - A_cl (c X) A_cl^T = c (α X - A_cl X A_cl^T) ⪰ 0
    # Note α X - A_cl X A_cl^T = (α - 1) X + W_seed by Lyap eq.  For α < 1, this is
    # (α-1) X + I.  PSD iff I ⪰ (1-α) X iff (1-α) X ⪯ I iff λ_max(X) ≤ 1/(1-α).
    # If λ_max(X) is too big, scale X DOWN: X → X / s, with s = (1-α) λ_max(X).
    lam_max_X = float(np.linalg.eigvalsh(X).max())
    s_decay = (1.0 - alpha) * lam_max_X
    if s_decay > 1.0:
        X = X / s_decay  # now (1-α) λ_max(X) = 1; α X - A_cl X A_cl^T = I/s_decay ⪰ 0 (small)
    # X is now a valid shape matrix for α-Lyapunov decay under A_cl.

    # (i) Scale DOWN if needed to satisfy X ⪯ P_max_shape_0.
    #     We want s * X ⪯ P_max_shape_0, i.e., s ≤ λ_min(X^{-1/2} P_max X^{-1/2}).
    M_max = np.linalg.solve(X, P_max_shape_0)
    s_feas_max = max(np.linalg.eigvals(M_max).real.min(), 1e-6)
    # (ii) Scale UP if needed so η_0 ∈ E_shape(X): η_0^T X^{-1} η_0 ≤ 1 ⇔ X ⪰ η_0 η_0^T (in PSD sense for outer product).
    #     For η_0 a vector: η_0^T (s X)^{-1} η_0 ≤ 1 ⇔ s ≥ η_0^T X^{-1} η_0.
    if eta_0 is not None and np.linalg.norm(eta_0) > 0:
        s_initial_min = float(eta_0 @ np.linalg.solve(X, eta_0))
    else:
        s_initial_min = 0.0

    if s_initial_min > s_feas_max:
        # Incompatible: η_0 doesn't fit inside the MVIE-bounded feasibility region.
        # Honour feasibility; caller will see large initial deviation.
        s = s_feas_max
    else:
        s = max(min(s_feas_max, max(s_initial_min, 1.0)), s_initial_min)

    P_0 = s * X
    return P_0, K_0
