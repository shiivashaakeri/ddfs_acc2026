"""
Persistence of Excitation Signals.

Generate bounded excitation signals to ensure data informativity
for the data-driven approach (Section IV-A in paper).

Requirements:
- ||epsilon(k)|| <= eps_bar for all k (bounded excitation)
- rank([H_i; Xi_i]) = n + m (persistence of excitation)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def generate_excitation(
    m: int,
    eps_bar: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate random bounded excitation signal.

    Samples uniformly from the ball ||epsilon|| <= eps_bar.

    Args:
        m: Input dimension (2 for 2-DoF arm)
        eps_bar: Bound on excitation norm (paper value: 0.15)
        rng: Random number generator (optional)

    Returns:
        epsilon: (m,) excitation signal with ||epsilon|| <= eps_bar
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample direction uniformly on unit sphere
    direction = rng.standard_normal(m)
    direction = direction / np.linalg.norm(direction)

    # Sample radius uniformly in [0, eps_bar]
    # Use r^(1/m) for uniform sampling in ball
    radius = eps_bar * rng.uniform() ** (1.0 / m)

    return radius * direction


def generate_excitation_sequence(
    m: int,
    L: int,
    eps_bar: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate sequence of bounded excitation signals.

    Args:
        m: Input dimension
        L: Sequence length
        eps_bar: Bound on excitation norm
        seed: Random seed for reproducibility

    Returns:
        epsilon_seq: (L, m) excitation sequence
    """
    rng = np.random.default_rng(seed)
    epsilon_seq = np.zeros((L, m))

    for k in range(L):
        epsilon_seq[k] = generate_excitation(m, eps_bar, rng)

    return epsilon_seq


def generate_sinusoidal_excitation(
    m: int,
    L: int,
    eps_bar: float,
    frequencies: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate sinusoidal excitation signals for better persistence of excitation.

    Sinusoidal signals provide richer frequency content than random noise,
    which can improve the rank condition.

    Args:
        m: Input dimension
        L: Sequence length
        eps_bar: Bound on excitation norm
        frequencies: (m,) array of frequencies (rad/sample). Default varies per channel.

    Returns:
        epsilon_seq: (L, m) excitation sequence
    """
    if frequencies is None:
        # Use different frequencies for each channel
        frequencies = np.array([0.1 + 0.05 * i for i in range(m)])

    epsilon_seq = np.zeros((L, m))
    t = np.arange(L)

    for i in range(m):
        # Generate sinusoid with varying phase
        phase = np.pi / 4 * i
        epsilon_seq[:, i] = eps_bar * np.sin(frequencies[i] * t + phase)

    # Ensure bound is satisfied at each time step
    for k in range(L):
        norm = np.linalg.norm(epsilon_seq[k])
        if norm > eps_bar:
            epsilon_seq[k] = epsilon_seq[k] * eps_bar / norm

    return epsilon_seq


def verify_pe_condition(
    H: np.ndarray,
    Xi: np.ndarray,
    n: int,
    m: int,
    tol: float = 1e-10,
) -> Tuple[bool, int]:
    """
    Verify persistence of excitation (PE) condition.

    The PE condition requires: rank([H; Xi]) = n + m

    This ensures the data matrices contain enough information
    for identifying the system dynamics.

    Args:
        H: (n, L) State deviation matrix
        Xi: (m, L) Input deviation matrix
        n: State dimension
        m: Input dimension
        tol: Tolerance for rank computation

    Returns:
        satisfied: True if rank([H; Xi]) = n + m
        rank: Computed rank
    """
    # Stack matrices vertically
    stacked = np.vstack([H, Xi])

    # Compute rank
    rank = np.linalg.matrix_rank(stacked, tol=tol)

    # Check if full rank
    required_rank = n + m
    satisfied = rank == required_rank

    return satisfied, rank


def compute_singular_values(H: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """
    Compute singular values of [H; Xi] for PE analysis.

    Small singular values indicate poor excitation in certain directions.

    Args:
        H: (n, L) State deviation matrix
        Xi: (m, L) Input deviation matrix

    Returns:
        singular_values: Singular values in descending order
    """
    stacked = np.vstack([H, Xi])
    _, s, _ = np.linalg.svd(stacked, full_matrices=False)
    return s


def estimate_excitation_quality(
    H: np.ndarray,
    Xi: np.ndarray,
    n: int,
    m: int,
) -> dict:
    """
    Estimate the quality of excitation in the data.

    Returns metrics about how well-conditioned the data is
    for system identification.

    Args:
        H: State deviation matrix
        Xi: Input deviation matrix
        n: State dimension
        m: Input dimension

    Returns:
        dict with:
            rank: Computed rank
            required_rank: n + m
            satisfied: Whether PE condition is met
            singular_values: All singular values
            condition_number: Ratio of largest to smallest singular value
            min_singular_value: Smallest singular value
    """
    satisfied, rank = verify_pe_condition(H, Xi, n, m)
    singular_values = compute_singular_values(H, Xi)

    # Compute condition number (handle zero singular values)
    min_sv = singular_values[-1] if len(singular_values) > 0 else 0
    max_sv = singular_values[0] if len(singular_values) > 0 else 0
    condition_number = max_sv / min_sv if min_sv > 1e-15 else np.inf

    return {
        "rank": rank,
        "required_rank": n + m,
        "satisfied": satisfied,
        "singular_values": singular_values,
        "condition_number": condition_number,
        "min_singular_value": min_sv,
    }


class ExcitationGenerator:
    """
    Generator for persistence of excitation signals.

    Provides consistent excitation generation with configurable parameters.
    """

    def __init__(
        self,
        m: int,
        eps_bar: float,
        seed: Optional[int] = None,
        mode: str = "random",
    ):
        """
        Initialize excitation generator.

        Args:
            m: Input dimension
            eps_bar: Excitation bound
            seed: Random seed
            mode: "random" or "sinusoidal"
        """
        self.m = m
        self.eps_bar = eps_bar
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self._step = 0

        # Frequencies for sinusoidal mode
        self.frequencies = np.array([0.1 + 0.05 * i for i in range(m)])

    def __call__(self) -> np.ndarray:
        """Generate single excitation signal."""
        if self.mode == "random":
            epsilon = generate_excitation(self.m, self.eps_bar, self.rng)
        elif self.mode == "sinusoidal":
            epsilon = np.zeros(self.m)
            for i in range(self.m):
                phase = np.pi / 4 * i
                epsilon[i] = self.eps_bar * np.sin(
                    self.frequencies[i] * self._step + phase
                )
            # Ensure bound
            norm = np.linalg.norm(epsilon)
            if norm > self.eps_bar:
                epsilon = epsilon * self.eps_bar / norm
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._step += 1
        return epsilon

    def reset(self, seed: Optional[int] = None):
        """Reset generator state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step = 0
