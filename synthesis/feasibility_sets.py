"""Maximum-volume inscribed ellipsoid (MVIE) bounds for feasibility (Remark 2).

For each time k along the nominal trajectory, the linearized state constraints
in deviation coordinates form a polytope:

    a_j^T η <= b_j(k),  with  a_j(k) = ∇h_j^x(x̂_nom(k)),  b_j(k) = -h_j^x(x̂_nom(k)).

For a simple box constraint x_min <= x <= x_max:
    each face i contributes  e_i^T η <= x_max,i - x̂_nom_i(k)
                            -e_i^T η <= x̂_nom_i(k) - x_min,i.

The maximum-volume inscribed ellipsoid is

    P_min(k)^{-1/2} = argmax log det Z   s.t.  ||Z a_j||_2 <= b_j(k), ∀j.

Then E(P_min(k)) ⊆ {η : x̂_nom(k) + η ∈ X}.

The same construction gives R_max(k) in ξ-coordinates around û_nom(k).

For each segment we take envelopes  P_min,i ⪰ max_k P_min(k)  and
R_max,i ⪯ min_k R_max(k) over k ∈ T_i (Section V.A).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def mvie_from_box(
    center: np.ndarray, lo: np.ndarray, hi: np.ndarray, verbose: bool = False
) -> np.ndarray:
    """MVIE inside the box {x : lo <= x <= hi} around `center`.

    Returns Z = P^{-1/2} ≻ 0 such that {η = Z u : ||u|| <= 1} is the
    largest ellipsoid contained in the shifted box.

    Equivalent characterization: P = Z^{-2} ≻ 0 and {η : η^T P η <= 1}
    is the MVIE. Returns Z; callers can compute P = (Z @ Z).inv().
    """
    n = center.size
    Z = cp.Variable((n, n), symmetric=True)
    # MVIE for an axis-aligned box reduces to Z being PSD with
    # ||Z e_i||_2 <= min(hi_i - c_i, c_i - lo_i) for each i.
    constraints = [Z >> 0]
    for i in range(n):
        slack = float(min(hi[i] - center[i], center[i] - lo[i]))
        if slack <= 0:
            raise ValueError(
                f"center component {i} = {center[i]} not strictly inside [{lo[i]}, {hi[i]}]"
            )
        constraints.append(cp.norm(Z[:, i], 2) <= slack)
    prob = cp.Problem(cp.Maximize(cp.log_det(Z)), constraints)
    prob.solve(verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"MVIE solve failed: {prob.status}")
    return np.asarray(Z.value)


def Pmin_from_box(center: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """V-matrix convention: returns P ≻ 0 such that {η : η^T P η ≤ 1} = MVIE in box."""
    Z = mvie_from_box(center, lo, hi)
    P = np.linalg.inv(Z @ Z)
    return 0.5 * (P + P.T)


def Pmax_shape_from_box(center: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Shape-matrix convention: returns P ≻ 0 such that {η : η^T P^{-1} η ≤ 1}
    is the maximum-volume ellipsoid inscribed in the shifted box.

    For an axis-aligned box with center c and face slacks d_i = min(hi_i-c_i, c_i-lo_i),
    this is diag(d_i^2). A funnel with shape P satisfies E_shape(P) ⊆ box iff P ⪯ this.
    """
    Z = mvie_from_box(center, lo, hi)
    P_shape = Z @ Z  # in our box case Z is diagonal, so this is diag(d_i^2)
    return 0.5 * (P_shape + P_shape.T)


def compute_Pmin_Rmax_along_nominal(
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
    shape_convention: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time feasibility ellipsoid envelopes along the nominal.

    shape_convention=True (default, matches paper's LMI):
        P[k] is the shape matrix of the MVIE in state-deviation box. A funnel
        with shape P_funnel satisfies E_shape(P_funnel) ⊆ box iff P_funnel ⪯ P[k].
    shape_convention=False:
        P[k] is the V-matrix convention (Def 4 as written). A funnel with V-matrix
        P_funnel satisfies E_V(P_funnel) ⊆ box iff P_funnel ⪰ P[k].

    R[k] is always in the shape convention (R = shape, E_u uses R^{-1}).
    """
    N = u_nom.shape[0]
    n = x_nom.shape[1]
    m = u_nom.shape[1]
    P_env = np.zeros((N + 1, n, n))
    R_env = np.zeros((N, m, m))
    for k in range(N + 1):
        if shape_convention:
            P_env[k] = Pmax_shape_from_box(x_nom[k], x_lo, x_hi)
        else:
            P_env[k] = Pmin_from_box(x_nom[k], x_lo, x_hi)
    for k in range(N):
        Z = mvie_from_box(u_nom[k], u_lo, u_hi)
        R = Z @ Z
        R_env[k] = 0.5 * (R + R.T)
    return P_env, R_env


def segment_envelopes(
    P_env_per_step: np.ndarray,
    R_env_per_step: np.ndarray,
    segment_indices: range,
    shape_convention: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Single per-segment envelopes for a contiguous range of time indices.

    For diagonal MVIE matrices (axis-aligned boxes), this reduces to elementwise
    min on the diagonals (shape conv: funnel ⪯ env on every k) or elementwise
    max (V conv).

    shape_convention=True:
        Returns (P_max, R_max) where funnel must satisfy P ⪯ P_max, R ⪯ R_max
        (both in shape convention).
    """
    seg_P = P_env_per_step[list(segment_indices)]
    seg_R = R_env_per_step[list(segment_indices)]
    if shape_convention:
        # Shape ellipsoid ⊆ box ⇔ P_funnel ⪯ P_box. Take pointwise MIN over the segment.
        P_envelope = np.min(np.abs(seg_P), axis=0)
    else:
        # V-matrix: ellipsoid ⊆ box ⇔ P_funnel ⪰ P_box. Take pointwise MAX.
        P_envelope = np.max(np.abs(seg_P), axis=0)
    P_envelope = 0.5 * (P_envelope + P_envelope.T)
    eigvals, V = np.linalg.eigh(P_envelope)
    eigvals = np.maximum(eigvals, eigvals.max() * 1e-12)
    P_envelope = V @ np.diag(eigvals) @ V.T

    R_envelope = np.min(np.abs(seg_R), axis=0)
    R_envelope = 0.5 * (R_envelope + R_envelope.T)
    eigvals, V = np.linalg.eigh(R_envelope)
    eigvals = np.maximum(eigvals, eigvals.max() * 1e-12)
    R_envelope = V @ np.diag(eigvals) @ V.T

    return P_envelope, R_envelope
