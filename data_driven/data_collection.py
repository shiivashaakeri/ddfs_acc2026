"""
Segment-wise Data Collection.

Manages segmented data collection and constructs data matrices
H_i, H_i^+, Xi_i for controller synthesis (Section IV-A).

Key indices (from paper):
- Segment i: T_i = {k_i, ..., k_{i+1} - 1} where k_i = i * T
- Data window: T_i^D = {k_{i+1} - L, ..., k_{i+1} - 1}
- With T=100, L=60: Segment 0 is k=0..99, data window is k=40..99
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class SegmentManager:
    """
    Manages segment indices and data windows for segmented control.

    The trajectory is divided into segments of length T. Each segment i
    has a data window T_i^D of length L at the end of the segment, used
    for data-driven uncertainty quantification.
    """

    def __init__(self, T: int, L: int, n: int, m: int):
        """
        Initialize segment manager.

        Args:
            T: Segment length (paper value: 100)
            L: Data window length (paper value: 60)
            n: State dimension (4 for 2-DoF arm)
            m: Input dimension (2 for 2-DoF arm)
        """
        if L > T:
            raise ValueError(f"Data window L={L} cannot exceed segment length T={T}")

        self.T = T
        self.L = L
        self.n = n
        self.m = m

    def num_segments(self, N: int) -> int:
        """
        Compute number of complete segments in trajectory of length N.

        Args:
            N: Total trajectory length

        Returns:
            Number of complete segments
        """
        return N // self.T

    def get_segment_start(self, i: int) -> int:
        """
        Get starting index k_i of segment i.

        Args:
            i: Segment index (0-based)

        Returns:
            k_i = i * T
        """
        return i * self.T

    def get_segment_end(self, i: int) -> int:
        """
        Get ending index k_{i+1} of segment i (exclusive).

        Args:
            i: Segment index (0-based)

        Returns:
            k_{i+1} = (i + 1) * T
        """
        return (i + 1) * self.T

    def get_segment_indices(self, i: int) -> range:
        """
        Get indices for segment i: T_i = {k_i, ..., k_{i+1} - 1}.

        Args:
            i: Segment index (0-based)

        Returns:
            range(k_i, k_{i+1})
        """
        k_i = self.get_segment_start(i)
        k_next = self.get_segment_end(i)
        return range(k_i, k_next)

    def get_data_window_indices(self, i: int) -> range:
        """
        Get indices for data window T_i^D = {k_{i+1} - L, ..., k_{i+1} - 1}.

        The data window is the last L steps of segment i.

        Args:
            i: Segment index (0-based)

        Returns:
            range(k_{i+1} - L, k_{i+1})
        """
        k_next = self.get_segment_end(i)
        return range(k_next - self.L, k_next)

    def get_data_window_start(self, i: int) -> int:
        """
        Get starting index of data window for segment i.

        Args:
            i: Segment index

        Returns:
            k_{i+1} - L
        """
        return self.get_segment_end(i) - self.L

    def is_in_segment(self, k: int, i: int) -> bool:
        """
        Check if time index k is in segment i.

        Args:
            k: Time index
            i: Segment index

        Returns:
            True if k in T_i
        """
        return k in self.get_segment_indices(i)

    def is_in_data_window(self, k: int, i: int) -> bool:
        """
        Check if time index k is in data window of segment i.

        Args:
            k: Time index
            i: Segment index

        Returns:
            True if k in T_i^D
        """
        return k in self.get_data_window_indices(i)

    def get_segment_for_index(self, k: int) -> int:
        """
        Get segment index containing time index k.

        Args:
            k: Time index

        Returns:
            Segment index i such that k in T_i
        """
        return k // self.T


def build_data_matrices(
    eta_log: np.ndarray,
    xi_log: np.ndarray,
    segment_i: int,
    segment_manager: SegmentManager,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build data matrices H_i, H_i^+, Xi_i from logged deviation data.

    From Eq. (9-10) in the paper:
    - H_i = [eta(k_i^D), ..., eta(k_{i+1}-1)]  (n x L)
    - H_i^+ = [eta(k_i^D+1), ..., eta(k_{i+1})]  (n x L)
    - Xi_i = [xi(k_i^D), ..., xi(k_{i+1}-1)]  (m x L)

    Args:
        eta_log: (N+1, n) State deviation log eta(k) = x(k) - x_nom(k)
        xi_log: (N, m) Input deviation log xi(k) = u(k) - u_nom(k)
        segment_i: Segment index
        segment_manager: SegmentManager instance

    Returns:
        H_i: (n, L) State deviation matrix
        H_i_plus: (n, L) Next-state deviation matrix
        Xi_i: (m, L) Input deviation matrix
    """
    data_window = segment_manager.get_data_window_indices(segment_i)
    L = segment_manager.L
    n = segment_manager.n
    m = segment_manager.m

    H_i = np.zeros((n, L))
    H_i_plus = np.zeros((n, L))
    Xi_i = np.zeros((m, L))

    for j, k in enumerate(data_window):
        H_i[:, j] = eta_log[k]
        H_i_plus[:, j] = eta_log[k + 1]
        Xi_i[:, j] = xi_log[k]

    return H_i, H_i_plus, Xi_i


def check_rank_condition(
    H_i: np.ndarray, Xi_i: np.ndarray, n: int, m: int, tol: float = 1e-10
) -> Tuple[bool, int]:
    """
    Check persistence of excitation rank condition (Eq. 11).

    The rank condition requires: rank([H_i; Xi_i]) = n + m

    This ensures the data is rich enough for identification.

    Args:
        H_i: (n, L) State deviation matrix
        Xi_i: (m, L) Input deviation matrix
        n: State dimension
        m: Input dimension
        tol: Tolerance for rank computation

    Returns:
        satisfied: True if rank condition is met
        rank: Computed rank of [H_i; Xi_i]
    """
    # Stack matrices vertically: [H_i; Xi_i] is (n+m, L)
    stacked = np.vstack([H_i, Xi_i])

    # Compute rank
    rank = np.linalg.matrix_rank(stacked, tol=tol)

    # Check if rank equals n + m
    satisfied = rank == n + m

    return satisfied, rank


def compute_T_tilde(segment_manager: SegmentManager) -> int:
    """
    Compute T̃ = 2T - L - 1, the maximum index difference in data-driven bounds.

    From the paper, T̃ bounds the Jacobian variation in the uncertainty set.

    Args:
        segment_manager: SegmentManager instance

    Returns:
        T_tilde = 2T - L - 1
    """
    return 2 * segment_manager.T - segment_manager.L - 1
