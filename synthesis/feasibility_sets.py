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
    """Returns P_min ≻ 0 such that E(P_min) ⊆ shifted box."""
    Z = mvie_from_box(center, lo, hi)
    # P = (Z Z)^{-1}, symmetrized
    P = np.linalg.inv(Z @ Z)
    return 0.5 * (P + P.T)


def compute_Pmin_Rmax_along_nominal(
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time MVIE ellipsoids P_min(k), R_max(k).

    Returns:
        P_min : (N+1, n, n)
        R_max : (N, m, m)
    """
    N = u_nom.shape[0]
    n = x_nom.shape[1]
    m = u_nom.shape[1]
    P_min = np.zeros((N + 1, n, n))
    R_max = np.zeros((N, m, m))
    for k in range(N + 1):
        P_min[k] = Pmin_from_box(x_nom[k], x_lo, x_hi)
    for k in range(N):
        # R_max plays the symmetric role for inputs (its inverse bounds the input ellipsoid).
        # Compute MVIE in ξ-space and store the corresponding R_max = Z Z (so E_u(R_max) = {ξ : ξ^T R_max^{-1} ξ <= 1}).
        Z = mvie_from_box(u_nom[k], u_lo, u_hi)
        R_max[k] = Z @ Z
        R_max[k] = 0.5 * (R_max[k] + R_max[k].T)
    return P_min, R_max


def segment_envelopes(
    P_min: np.ndarray,
    R_max: np.ndarray,
    segment_indices: range,
) -> tuple[np.ndarray, np.ndarray]:
    """Single per-segment envelopes (Section V.A).

    P_min,i ⪰ max_k P_min(k)  -- we use the largest eigenvalue version:
        pick P_min,i = max scalar c s.t. c I ⪰ P_min(k) for all k in segment;
        but simpler & paper-compatible: take the pointwise max of P_min via a
        common diagonal upper bound. For axis-aligned boxes the MVIE is diagonal
        in expectation, so the elementwise max of the diagonals works.

    R_max,i ⪯ min_k R_max(k)  -- elementwise min of the diagonals.
    """
    seg_P = P_min[list(segment_indices)]
    seg_R = R_max[list(segment_indices)]
    # Take pointwise max of P (most restrictive ellipsoid -- smallest)
    # by taking entrywise max along the diagonal in the eigenbasis. For our
    # diagonal-ish boxes the simple per-element max on the full matrix is a
    # valid PSD upper bound; we then symmetrize. This is conservative but
    # correct.
    P_envelope = np.max(np.abs(seg_P), axis=0)
    P_envelope = 0.5 * (P_envelope + P_envelope.T)
    # Make sure it's PSD by symmetrizing + tiny eigenvalue projection
    eigvals, V = np.linalg.eigh(P_envelope)
    eigvals = np.maximum(eigvals, eigvals.max() * 1e-12)
    P_envelope = V @ np.diag(eigvals) @ V.T

    R_envelope = np.min(np.abs(seg_R), axis=0)
    R_envelope = 0.5 * (R_envelope + R_envelope.T)
    eigvals, V = np.linalg.eigh(R_envelope)
    eigvals = np.maximum(eigvals, eigvals.max() * 1e-12)
    R_envelope = V @ np.diag(eigvals) @ V.T

    return P_envelope, R_envelope
