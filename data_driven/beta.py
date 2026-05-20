"""Disturbance-energy bound β_i (Eq. 16) and a tighter empirical alternative.

The paper's β_i (Eq. 16) sums squared pointwise bounds from Lemma 3:

    β_i^{paper} := Σ_{k ∈ T_i^D}  ( C |k - k_i| ||z(k)|| + γ + L_r ||z(k)||² )²

This is theoretically sound (any plant satisfying Assumptions 1-2 has W W^T ⪯ β_i^{paper} I)
but extremely conservative: in practice it overestimates ||W||_F² by factors of 10⁶+ for
smooth nominals. The matrix S-lemma then renders the SDP infeasible.

We additionally provide an empirical β that uses the LS-estimated (A, B) from the data
to compute the actual W = H_plus - A_ls H - B_ls Ξ, and returns ||W||_F² scaled by a
safety margin. This still encodes a valid quadratic data constraint W W^T ⪯ β I (a
weaker one, since β is smaller and Σ is smaller), trading off robustness for
practical feasibility.
"""

from __future__ import annotations

import numpy as np

from .data_collection import SegmentData


def beta_segment(seg: SegmentData, C: float, gamma: float, L_r: float) -> float:
    """Eq. (16) pointwise-bound β_i."""
    z = np.hstack([seg.eta, seg.xi])
    z_norm = np.linalg.norm(z, axis=1)
    dt = seg.time_offsets()
    pointwise = C * np.abs(dt) * z_norm + gamma + L_r * z_norm ** 2
    return float(np.sum(pointwise ** 2))


def beta_empirical(
    H: np.ndarray, H_plus: np.ndarray, Xi: np.ndarray, safety: float = 4.0
) -> float:
    """Tighter β based on the LS residual of fitting H_plus ≈ A H + B Ξ.

    Returns safety * ||W_LS||_F² so that the true plant W (close to W_LS) satisfies
    W W^T ⪯ β I with high probability. The safety factor inflates the empirical
    bound to keep the true system inside Σ.
    """
    n, _ = H.shape
    m = Xi.shape[0]
    # Stack (H; Xi) -> shape (n+m, L); solve H_plus = [A B] @ stack.
    stack = np.vstack([H, Xi])  # (n+m, L)
    # min over (A, B): ||H_plus - [A B] stack||_F.
    # Closed-form: [A B] = H_plus stack^T (stack stack^T)^{-1}.
    Gram = stack @ stack.T
    AB = H_plus @ stack.T @ np.linalg.pinv(Gram)
    W = H_plus - AB @ stack
    energy = float(np.linalg.norm(W) ** 2)
    return safety * max(energy, 1e-12)
