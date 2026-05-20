"""Data collection on segments (Section IV.A, Eqs. 7–11).

During segment T_i the controller applies ξ(k) = K_i η(k). During the
trailing data window T_i^D = {k_{i+1}-L, ..., k_{i+1}-1}, a bounded
excitation ε(k) is added: ξ(k) = K_i η(k) + ε(k).

At the end of the segment we assemble:

    H_i      = [η(k_i^D)   ... η(k_{i+1}-1)]     ∈ ℝ^{n × L}
    H_i^+    = [η(k_i^D+1) ... η(k_{i+1})  ]     ∈ ℝ^{n × L}
    Ξ_i      = [ξ(k_i^D)   ... ξ(k_{i+1}-1)]     ∈ ℝ^{m × L}

Persistence-of-excitation requires rank([H_i; Ξ_i]) = n + m (Eq. 11).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundedExcitation:
    """Bounded random excitation generator (||ε|| ≤ ε̄)."""

    eps_bar: float
    dim: int
    rng: np.random.Generator

    def sample(self) -> np.ndarray:
        v = self.rng.standard_normal(self.dim)
        v /= max(np.linalg.norm(v), 1e-12)
        # Magnitude uniform in (0, ε̄]: sample uniformly in the ball.
        mag = self.eps_bar * self.rng.uniform(0.0, 1.0) ** (1.0 / self.dim)
        return mag * v

    def sequence(self, L: int) -> np.ndarray:
        return np.stack([self.sample() for _ in range(L)])


@dataclass
class SegmentData:
    """Raw logged tuples (η_k, ξ_k, η_{k+1}) over a segment's data window."""

    eta: np.ndarray  # (L, n)  -- η(k) for k in T_i^D
    xi: np.ndarray  # (L, m)
    eta_next: np.ndarray  # (L, n)  -- η(k+1)
    k_start: int  # k_i^D
    k_segment_start: int  # k_i

    @property
    def L(self) -> int:
        return self.eta.shape[0]

    def time_offsets(self) -> np.ndarray:
        """|k - k_i| for each row, used in the β bound."""
        return np.arange(self.k_start, self.k_start + self.L) - self.k_segment_start


def assemble_data_matrices(seg: SegmentData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (H, H_plus, Xi) of shapes (n, L), (n, L), (m, L)."""
    H = seg.eta.T
    H_plus = seg.eta_next.T
    Xi = seg.xi.T
    return H, H_plus, Xi


def persistently_exciting(H: np.ndarray, Xi: np.ndarray, tol: float = 1e-8) -> bool:
    """Rank check (Eq. 11): rank([H; Xi]) == n + m."""
    stacked = np.vstack([H, Xi])
    n_plus_m = H.shape[0] + Xi.shape[0]
    s = np.linalg.svd(stacked, compute_uv=False)
    return int(np.sum(s > tol * s[0])) == n_plus_m
