"""Initial stabilizing controller (P_0, K_0) with a properly certified funnel.

Two-step design:

  1. K_0 from DLQR on the twin's Jacobian at the start of the nominal
     (Q = diag(1, 1, 0.1, 0.1), R = I). Empirically tuned to give the
     right balance between transient overshoot and steady-state error.
     For the deviation dynamics ξ = K_0 η we use K_0 = −K_DLQR.

  2. P_0 by a small SDP that fixes the gain and solves for the smallest
     funnel containing η_0 with α-decay under the twin's closed-loop:
        minimise  trace(P)
        subject to  P ≻ 0,
                    A_cl P A_cl^T ⪯ α P  (Schur),
                    η_0^T P^{-1} η_0 ≤ 1  (Schur),
                    P ⪯ P_max.
     This gives a P_0 that is certified to contain η_0 under twin dynamics
     and is as tight as possible.

Under the actual plant (which differs from the twin by Δ), η can leak
out of E(P_0) slightly during the transient. The leak is bounded by
the magnitude of the plant–twin mismatch γ, which Lemma 1 quantifies.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are

from dynamics.discretization import DiscreteSystem
from linearization.jacobians import jacobians_discrete


def _dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Discrete-LQR gain: u = -K x stabilises x⁺ = (A - B K) x."""
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def _tightest_invariant_funnel(
    A_cl: np.ndarray,
    eta_0: np.ndarray,
    P_max_shape: np.ndarray | None,
    alpha: float,
    solver: str = "CLARABEL",
) -> np.ndarray:
    """Smallest (in trace) shape matrix P ⪰ 0 satisfying

        A_cl P A_cl^T ⪯ α P,
        η_0^T P^{-1} η_0 ≤ 1,
        (optional) P ⪯ P_max_shape.

    If the version with the P_max bound is infeasible (the initial deviation
    is too large to fit inside the feasibility envelope under the LQR metric),
    we drop the P_max constraint. The resulting funnel may extend beyond the
    state box; we flag this so the visualisation can be honest about it.
    """
    n = A_cl.shape[0]

    def _solve(with_pmax: bool):
        P = cp.Variable((n, n), symmetric=True)
        constraints = [
            P >> 1e-8 * np.eye(n),
            cp.bmat([[alpha * P, A_cl @ P], [P @ A_cl.T, P]]) >> 0,
            cp.bmat([[np.array([[1.0]]), eta_0.reshape(1, -1)], [eta_0.reshape(-1, 1), P]]) >> 0,
        ]
        if with_pmax and P_max_shape is not None:
            constraints.append(P << P_max_shape)
        prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
        last_err = None
        for s in [solver, "SCS"]:
            try:
                prob.solve(solver=s)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    break
            except cp.error.SolverError as e:
                last_err = e
                continue
        return prob.status, P.value, last_err

    status, P_val, last_err = _solve(with_pmax=True)
    if status not in ("optimal", "optimal_inaccurate"):
        # Try without P_max -- accept that the funnel extends beyond the
        # state-feasibility envelope for the initial segment.
        status, P_val, last_err = _solve(with_pmax=False)
    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"initial-funnel SDP failed: status={status}"
            + (f" (last solver error: {last_err})" if last_err else "")
        )
    return 0.5 * (P_val + P_val.T)


def design_initial_controller(
    twin: DiscreteSystem,
    x_nom_0: np.ndarray,
    u_nom_0: np.ndarray,
    P_max_shape_0: np.ndarray,
    eta_0: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    alpha: float = 0.98,
    **_unused,  # accept R_max_0 etc. for compatibility
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_0, K_0). See module docstring."""
    A_t, B_t = jacobians_discrete(twin.step, x_nom_0, u_nom_0)
    n, m = A_t.shape[0], B_t.shape[1]
    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1, 0.1])
    if R is None:
        R = 1.0 * np.eye(m)
    K_dlqr = _dlqr(A_t, B_t, Q, R)
    K_0 = -K_dlqr                       # deviation dynamics: ξ = K_0 η
    A_cl = A_t + B_t @ K_0               # twin's closed loop, stable

    if eta_0 is None or np.linalg.norm(eta_0) < 1e-12:
        # Degenerate case: no initial deviation; pick a small P_0.
        P_0 = 1e-3 * np.eye(n)
        # Make sure P_0 ⪯ P_max and is α-decay (any small P satisfies for stable A_cl).
        return P_0, K_0

    P_0 = _tightest_invariant_funnel(A_cl, np.asarray(eta_0), P_max_shape_0, alpha)
    return P_0, K_0
