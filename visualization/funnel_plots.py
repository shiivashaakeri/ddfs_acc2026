"""
Ellipsoid and Funnel Visualization.

Visualize ellipsoidal funnels in 2D projections.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_ellipsoid_2d(
    P: np.ndarray,
    center: np.ndarray,
    ax: plt.Axes,
    indices: Tuple[int, int] = (0, 1),
    n_points: int = 100,
    **kwargs,
) -> mpatches.Ellipse:
    """
    Plot 2D projection of ellipsoid {x: (x-c)^T P (x-c) <= 1}.

    Args:
        P: (n, n) positive definite matrix
        center: (n,) center point
        ax: matplotlib axes
        indices: tuple of state indices to project onto (i, j)
        n_points: number of points for ellipse
        **kwargs: passed to matplotlib Ellipse

    Returns:
        ellipse: matplotlib Ellipse patch
    """
    i, j = indices
    center = np.asarray(center)

    # Extract 2x2 submatrix for the projection
    P_sub = np.array([
        [P[i, i], P[i, j]],
        [P[j, i], P[j, j]],
    ])

    # Compute ellipse parameters
    # Ellipse: x^T P x = 1
    # Eigendecomposition: P = V @ D @ V^T
    eigvals, eigvecs = np.linalg.eigh(P_sub)

    # Semi-axes: 1/sqrt(eigenvalues)
    semi_axes = 1.0 / np.sqrt(eigvals)

    # Rotation angle
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Default styling
    default_kwargs = {
        "fill": False,
        "edgecolor": "blue",
        "linewidth": 1,
        "alpha": 0.7,
    }
    default_kwargs.update(kwargs)

    ellipse = mpatches.Ellipse(
        xy=(center[i], center[j]),
        width=2 * semi_axes[0],
        height=2 * semi_axes[1],
        angle=angle,
        **default_kwargs,
    )
    ax.add_patch(ellipse)

    return ellipse


def plot_funnel_sequence(
    P_history: List[np.ndarray],
    x_nom: np.ndarray,
    state_indices: Tuple[int, int] = (0, 1),
    T: int = 100,
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    alpha: float = 0.3,
) -> plt.Figure:
    """
    Plot sequence of ellipsoids along nominal trajectory.

    Args:
        P_history: list of P matrices (one per segment)
        x_nom: (N+1, n) nominal trajectory
        state_indices: tuple of state indices to project onto
        T: segment length
        ax: matplotlib axes (created if None)
        colors: list of colors for each segment
        alpha: transparency

    Returns:
        fig: matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    i, j = state_indices
    N = x_nom.shape[0] - 1

    # Default colors
    if colors is None:
        cmap = plt.cm.viridis
        colors = [cmap(k / len(P_history)) for k in range(len(P_history))]

    # Plot nominal trajectory
    ax.plot(x_nom[:, i], x_nom[:, j], "k-", linewidth=2, label="Nominal", zorder=10)
    ax.plot(x_nom[0, i], x_nom[0, j], "go", markersize=10, label="Start", zorder=11)
    ax.plot(x_nom[-1, i], x_nom[-1, j], "r*", markersize=15, label="Goal", zorder=11)

    # Plot ellipsoids at segment boundaries
    for seg_idx, P in enumerate(P_history):
        seg_start = seg_idx * T
        if seg_start >= N:
            break

        center = x_nom[seg_start]
        plot_ellipsoid_2d(
            P, center, ax,
            indices=state_indices,
            edgecolor=colors[seg_idx],
            fill=True,
            facecolor=colors[seg_idx],
            alpha=alpha,
            linewidth=1.5,
        )

        # Also plot at segment midpoint
        seg_mid = min(seg_start + T // 2, N)
        center_mid = x_nom[seg_mid]
        plot_ellipsoid_2d(
            P, center_mid, ax,
            indices=state_indices,
            edgecolor=colors[seg_idx],
            fill=False,
            alpha=alpha * 1.5,
            linewidth=1,
            linestyle="--",
        )

    ax.set_xlabel(f"State {i}")
    ax.set_ylabel(f"State {j}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    return fig


def plot_funnel_evolution(
    P_history: List[np.ndarray],
    state_idx: int = 0,
    T: int = 100,
    dt: float = 0.01,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot evolution of funnel size over time for a single state.

    Args:
        P_history: list of P matrices
        state_idx: which state to plot
        T: segment length
        dt: time step
        ax: matplotlib axes

    Returns:
        fig: matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Extract diagonal element (funnel half-width = 1/sqrt(P[i,i]))
    times = []
    half_widths = []

    for seg_idx, P in enumerate(P_history):
        seg_start = seg_idx * T
        seg_end = (seg_idx + 1) * T

        if P[state_idx, state_idx] > 0:
            hw = 1.0 / np.sqrt(P[state_idx, state_idx])
        else:
            hw = np.inf

        times.extend([seg_start * dt, (seg_end - 1) * dt])
        half_widths.extend([hw, hw])

    ax.step(times, half_widths, where="post", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Funnel half-width (state {state_idx})")
    ax.set_title("Funnel Size Evolution")
    ax.grid(True, alpha=0.3)

    return fig


def plot_phase_portrait_with_funnel(
    x_trajectory: np.ndarray,
    x_nom: np.ndarray,
    P_history: List[np.ndarray],
    state_indices: Tuple[int, int] = (0, 1),
    T: int = 100,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot phase portrait with trajectory and funnels.

    Args:
        x_trajectory: (N+1, n) actual trajectory
        x_nom: (N+1, n) nominal trajectory
        P_history: list of P matrices
        state_indices: which states to plot
        T: segment length
        save_path: optional path to save figure

    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    i, j = state_indices

    # Plot funnels
    plot_funnel_sequence(
        P_history, x_nom,
        state_indices=state_indices,
        T=T,
        ax=ax,
        alpha=0.2,
    )

    # Plot actual trajectory
    ax.plot(
        x_trajectory[:, i], x_trajectory[:, j],
        "r-", linewidth=1.5, alpha=0.8, label="Actual"
    )

    ax.set_title("Phase Portrait with Funnels")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
