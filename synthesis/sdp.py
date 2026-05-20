"""Funnel synthesis SDP (Eq. 22).

For segment i+1, given the previous segment's uncertainty bundle and the
per-segment feasibility envelopes (P_min,i+1, R_max,i+1) and (optionally) the
previous funnel matrix P_i with cross-segment growth μ:

    maximize    log det(P)
    subject to  P ≻ 0,
                λ_1, λ_2 ≥ 0,
                ν > 0,
                S(P, L, ν) − λ_1 Ñ_1 − λ_2 Ñ_2 ⪰ 0,           (Theorem 1)
                P ⪰ P_min,i+1,                                (state feasibility)
                [[R_max,i+1, L]; [L^T, P]] ⪰ 0,               (input feasibility)
                P ⪯ μ P_i.                                    (cross-segment growth)

Recover K = L P^{-1}.

There are intentionally no regularizers in the objective or constraints.
If the SDP is infeasible at a segment, that's a real signal that the data
window did not deliver enough information for a certified controller, and
the right response is upstream (more excitation, longer L, smaller α, ...),
not a numerical band-aid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from data_driven.uncertainty import SegmentUncertainty
from synthesis.lmi_blocks import build_N1_tilde, build_N2_tilde, build_S_matrix


@dataclass
class FunnelDesign:
    P: np.ndarray
    K: np.ndarray
    L: np.ndarray
    nu: float
    lambda1: float
    lambda2: float
    log_det_P: float


def synthesize_funnel(
    bundle: SegmentUncertainty,
    P_min: np.ndarray,
    R_max: np.ndarray,
    alpha: float,
    P_prev: np.ndarray | None = None,
    mu: float | None = None,
    nu_lower: float = 1e-6,
    eig_lower: float = 1e-8,
    solver: str = "SCS",
    verbose: bool = False,
) -> FunnelDesign:
    """Solve Eq. (22) for one segment.

    Parameters
    ----------
    bundle
        SegmentUncertainty with (H, H_plus, Xi, beta, C, T_tilde) from segment i.
    P_min, R_max
        Per-segment feasibility envelopes for segment i+1.
    alpha
        Lyapunov decay rate, α ∈ (0, 1).
    P_prev, mu
        If both supplied, add the cross-segment growth constraint P ⪯ μ P_prev.
    nu_lower
        Strict positivity floor for ν.
    eig_lower
        Tiny PSD slack to keep ≻ from collapsing to a singular boundary.
    """
    n, m = bundle.n, bundle.m

    P = cp.Variable((n, n), symmetric=True)
    L_var = cp.Variable((m, n))
    nu = cp.Variable()
    lam1 = cp.Variable(nonneg=True)
    lam2 = cp.Variable(nonneg=True)

    S_expr = build_S_matrix(P, L_var, nu, alpha, n, m)
    N1_tilde = build_N1_tilde(bundle.H, bundle.H_plus, bundle.Xi, bundle.beta, n, m)
    N2_tilde = build_N2_tilde(bundle.C, bundle.T_tilde, n, m)

    constraints = [
        P >> eig_lower * np.eye(n),
        nu >= nu_lower,
        S_expr - lam1 * N1_tilde - lam2 * N2_tilde >> eig_lower * np.eye(S_expr.shape[0]),
        P >> P_min,
        cp.bmat([[R_max, L_var], [L_var.T, P]]) >> 0,
    ]
    if P_prev is not None and mu is not None:
        constraints.append(P << mu * P_prev)

    objective = cp.Maximize(cp.log_det(P))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"funnel SDP failed: status={prob.status}")

    P_val = np.asarray(P.value)
    P_val = 0.5 * (P_val + P_val.T)
    L_val = np.asarray(L_var.value)
    K_val = L_val @ np.linalg.inv(P_val)

    return FunnelDesign(
        P=P_val,
        K=K_val,
        L=L_val,
        nu=float(nu.value),
        lambda1=float(lam1.value),
        lambda2=float(lam2.value),
        log_det_P=float(np.log(np.linalg.det(P_val))),
    )
