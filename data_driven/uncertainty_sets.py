"""
Uncertainty Set Construction.

Define uncertainty set parameters for LMI-based funnel synthesis
from Definitions 2 and 3 in the paper.

The uncertainty sets capture:
1. Sigma_i: Jacobian uncertainty from time-varying linearization
2. Sigma_i^Delta: Plant-twin mismatch and nonlinear remainder

These sets are parameterized by:
- C² T̃²: Jacobian variation bound
- beta_i: Disturbance energy bound from data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class UncertaintyParams:
    """
    Container for uncertainty parameters used in LMI synthesis.

    Attributes:
        beta: Disturbance energy bound from Eq. (16)
        C2_T2: C² · T̃² bound for Jacobian variation
        C: Jacobian variation rate bound
        T_tilde: Maximum index difference (2T - L - 1)
        gamma: Plant-twin mismatch bound
        L_r: Second-order remainder constant
    """

    beta: float
    C2_T2: float
    C: float
    T_tilde: int
    gamma: float
    L_r: float

    def __post_init__(self):
        """Validate parameters are positive."""
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.C2_T2 < 0:
            raise ValueError(f"C2_T2 must be non-negative, got {self.C2_T2}")
        if self.C < 0:
            raise ValueError(f"C must be non-negative, got {self.C}")
        if self.T_tilde < 0:
            raise ValueError(f"T_tilde must be non-negative, got {self.T_tilde}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.L_r < 0:
            raise ValueError(f"L_r must be non-negative, got {self.L_r}")


def compute_jacobian_variation_bound(C: float, T_tilde: int) -> float:
    """
    Compute C² · T̃² for Jacobian variation bound in LMI.

    From Assumption 3 and Eq. (13), this bounds the squared norm
    of Jacobian differences over the segment.

    Args:
        C: Jacobian variation rate (C = L_J · v)
        T_tilde: Maximum index difference (2T - L - 1)

    Returns:
        C² · T̃²
    """
    return C**2 * T_tilde**2


def compute_T_tilde(T: int, L: int) -> int:
    """
    Compute T̃ = 2T - L - 1.

    This is the maximum value of |k - k_i| for k in the data window
    of the next segment, relative to the start of the current segment.

    Args:
        T: Segment length
        L: Data window length

    Returns:
        T_tilde = 2T - L - 1
    """
    return 2 * T - L - 1


def create_uncertainty_params(
    beta: float,
    C: float,
    T: int,
    L: int,
    gamma: float,
    L_r: float,
) -> UncertaintyParams:
    """
    Create UncertaintyParams from basic parameters.

    Args:
        beta: Computed disturbance energy bound
        C: Jacobian variation rate (C = L_J · v)
        T: Segment length
        L: Data window length
        gamma: Plant-twin mismatch bound
        L_r: Second-order remainder constant

    Returns:
        UncertaintyParams instance
    """
    T_tilde = compute_T_tilde(T, L)
    C2_T2 = compute_jacobian_variation_bound(C, T_tilde)

    return UncertaintyParams(
        beta=beta,
        C2_T2=C2_T2,
        C=C,
        T_tilde=T_tilde,
        gamma=gamma,
        L_r=L_r,
    )


@dataclass
class JacobianUncertaintySet:
    """
    Sigma_i: Jacobian uncertainty set (Definition 2).

    Represents the set of admissible (A, B) pairs given data.
    Parameterized by the data matrices H, H+, Xi and the
    Jacobian variation bound C² T̃².
    """

    H: np.ndarray  # (n, L) state deviation matrix
    H_plus: np.ndarray  # (n, L) next-state deviation matrix
    Xi: np.ndarray  # (m, L) input deviation matrix
    C2_T2: float  # Jacobian variation bound

    @property
    def n(self) -> int:
        """State dimension."""
        return self.H.shape[0]

    @property
    def m(self) -> int:
        """Input dimension."""
        return self.Xi.shape[0]

    @property
    def L(self) -> int:
        """Data window length."""
        return self.H.shape[1]


@dataclass
class DisturbanceUncertaintySet:
    """
    Sigma_i^Delta: Disturbance uncertainty set (Definition 3).

    Represents the set of admissible disturbance sequences
    bounded by beta_i.
    """

    beta: float  # Disturbance energy bound

    def contains(self, d_sequence: np.ndarray) -> bool:
        """
        Check if disturbance sequence is in the set.

        Args:
            d_sequence: (L, n) sequence of disturbances

        Returns:
            True if sum ||d(k)||² <= beta
        """
        energy = np.sum(np.linalg.norm(d_sequence, axis=1) ** 2)
        return energy <= self.beta


def compute_uncertainty_for_segment(
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    beta: float,
    C: float,
    T_tilde: int,
) -> tuple[JacobianUncertaintySet, DisturbanceUncertaintySet]:
    """
    Compute both uncertainty sets for a segment.

    Args:
        H: State deviation matrix
        H_plus: Next-state deviation matrix
        Xi: Input deviation matrix
        beta: Disturbance energy bound
        C: Jacobian variation rate
        T_tilde: Maximum index difference

    Returns:
        jacobian_set: JacobianUncertaintySet (Sigma_i)
        disturbance_set: DisturbanceUncertaintySet (Sigma_i^Delta)
    """
    C2_T2 = compute_jacobian_variation_bound(C, T_tilde)

    jacobian_set = JacobianUncertaintySet(
        H=H,
        H_plus=H_plus,
        Xi=Xi,
        C2_T2=C2_T2,
    )

    disturbance_set = DisturbanceUncertaintySet(beta=beta)

    return jacobian_set, disturbance_set
