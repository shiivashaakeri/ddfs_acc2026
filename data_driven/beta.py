"""Disturbance-energy bound β_i (Eq. 16).

Lemma 3 gives the pointwise bound

    ||w_i(k)|| ≤ C |k - k_i| ||[η(k); ξ(k)]|| + γ + L_r ||[η(k); ξ(k)]||².

Summing the squared pointwise bounds over the data window yields

    β_i := Σ_{k ∈ T_i^D}  ( C |k - k_i| ||z(k)|| + γ + L_r ||z(k)||² )²

where z(k) := [η(k); ξ(k)]. This is the constant used in the quadratic
data-consistency constraint W_i W_i^T ⪯ β_i I  (Definition 3, Eq. 15).
"""

from __future__ import annotations

import numpy as np

from .data_collection import SegmentData


def beta_segment(seg: SegmentData, C: float, gamma: float, L_r: float) -> float:
    """Compute β_i for one segment's data window."""
    z = np.hstack([seg.eta, seg.xi])  # (L, n+m)
    z_norm = np.linalg.norm(z, axis=1)  # (L,)
    dt = seg.time_offsets()
    pointwise = C * np.abs(dt) * z_norm + gamma + L_r * z_norm ** 2
    return float(np.sum(pointwise ** 2))
