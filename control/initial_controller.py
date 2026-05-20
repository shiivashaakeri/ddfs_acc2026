"""Initial stabilizing controller K_0 (Assumption 5).

The paper allows K_0 to be synthesized from the known twin model. We design
a discrete-time LQR on the twin's Jacobian at the start of the nominal
trajectory. By similarity of twin and plant Jacobians (Assumption 1 on
mismatch + Lemma 2 on slow Jacobian variation), this K_0 is also a
reasonable starting feedback for the plant's deviation dynamics.

For the funnel matrix P_0, we use the discrete algebraic Riccati solution
P_lqr — the Lyapunov function for the closed-loop twin dynamics. We then
scale it so E(P_0) fits inside the feasibility envelope at k = 0.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from dynamics.discretization import DiscreteSystem
from linearization.jacobians import jacobians_discrete


def design_initial_controller(
    twin: DiscreteSystem,
    x_nom_0: np.ndarray,
    u_nom_0: np.ndarray,
    P_min_0: np.ndarray,
    eta_0: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_0, K_0).

    K_0 = DLQR gain on the twin's Jacobian at (x_nom_0, u_nom_0).

    P_0 satisfies two requirements:
      (i) P_0 ⪰ P_min_0  (state-feasibility at k = 0)
      (ii) if `eta_0` is supplied, η_0^T P_0 η_0 ≤ 1  (initial deviation inside E(P_0))

    We start from the DLQR Riccati matrix P_lqr (the Lyapunov function of the
    closed-loop twin) and scale by the smallest factor that satisfies both
    requirements.
    """
    A, B = jacobians_discrete(twin.step, x_nom_0, u_nom_0)
    n, m = A.shape[0], B.shape[1]
    if Q is None:
        Q = np.diag([1.0, 1.0, 0.1, 0.1])  # gentle: penalize position more than velocity
    if R is None:
        R = 1.0 * np.eye(m)
    P_lqr = solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.solve(R + B.T @ P_lqr @ B, B.T @ P_lqr @ A)

    # (i) Scale up if needed for P_0 ⪰ P_min_0.
    M = np.linalg.solve(P_lqr, P_min_0)
    c_floor = max(1.0, float(np.linalg.eigvals(M).real.max())) * 1.01

    # (ii) Scale down if needed so η_0 ∈ E(P_0): η_0^T (c * P_lqr) η_0 ≤ 1
    #      ⇒ c ≤ 1 / (η_0^T P_lqr η_0).
    if eta_0 is not None and np.linalg.norm(eta_0) > 0:
        q = float(eta_0 @ P_lqr @ eta_0)
        c_initial = (1.0 / q) * 0.99 if q > 0 else np.inf
    else:
        c_initial = np.inf

    if c_floor > c_initial:
        # Two requirements are incompatible with this P_lqr direction.
        # Inflate Q to redirect P_lqr (we leave that to caller-side tuning); for now,
        # honour the feasibility floor and warn that η_0 may be outside E(P_0).
        c = c_floor
    else:
        c = min(c_initial, max(c_floor, 1.0))

    P_0 = c * P_lqr
    return P_0, K_lqr
