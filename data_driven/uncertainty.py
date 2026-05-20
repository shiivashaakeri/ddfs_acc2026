"""Bundled uncertainty description for one segment (Definitions 2-3).

Carries everything the funnel-synthesis LMI needs:

  Σ_i  : (A, B) consistent with H, H_plus, Xi via H_plus = A H + B Xi + W,
         W W^T ⪯ β I.
  Σ_i^Δ : Jacobian variation set, [ΔA ΔB][ΔA;ΔB]^T ⪯ C² T̃² I.

The synthesis module consumes a SegmentUncertainty without needing to know
how it was assembled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentUncertainty:
    """All data and bounds the funnel SDP needs for the *next* segment.

    By Algorithm 1 these come from segment i and are consumed when designing
    (P_{i+1}, K_{i+1}).
    """

    H: np.ndarray         # (n, L)
    H_plus: np.ndarray    # (n, L)
    Xi: np.ndarray        # (m, L)
    beta: float           # β_i
    C: float              # = L_J * v
    T_tilde: float        # bound on |k - k_i| over T_i ∪ T_{i+1}

    @property
    def n(self) -> int:
        return self.H.shape[0]

    @property
    def m(self) -> int:
        return self.Xi.shape[0]

    @property
    def L(self) -> int:
        return self.H.shape[1]
