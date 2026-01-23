"""
Trajectory Plots (Figure 4 Recreation).

Create 4-panel figure showing q1, q2, q1_dot, q2_dot tracking performance.

Elements:
- Black: nominal trajectory (twin)
- Red: proposed data-driven controller (plant)
- Gray dashed: baseline (twin controller on plant)
- Gray shaded: certified funnels
- Green: state constraints
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config.params import STATE_BOUNDS, SIM_PARAMS


def plot_tracking_results(
    results,
    baseline_results,
    dt: float = 0.01,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create Figure 4 from the paper: 4-panel tracking comparison.

    Args:
        results: SimulationResults from proposed method
        baseline_results: BaselineResults from baseline
        dt: Time step for x-axis
        save_path: Optional path to save figure

    Returns:
        fig: matplotlib Figure
    """
    N = results.x_trajectory.shape[0] - 1
    t = np.arange(N + 1) * dt

    # State labels and bounds
    state_labels = [r"$q_1$ [rad]", r"$q_2$ [rad]", r"$\dot{q}_1$ [rad/s]", r"$\dot{q}_2$ [rad/s]"]
    bounds_keys = ["q1", "q2", "dq1", "dq2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for i, (ax, label, bkey) in enumerate(zip(axes, state_labels, bounds_keys)):
        plot_single_state(
            ax=ax,
            t=t,
            x_nom=results.x_nom[:, i],
            x_proposed=results.x_trajectory[:, i],
            x_baseline=baseline_results.x_trajectory[:, i],
            bounds=STATE_BOUNDS[bkey],
            P_history=results.P_history,
            state_idx=i,
            T=SIM_PARAMS["T"],
            dt=dt,
            label=label,
        )

    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))

    # Common x-label
    for ax in axes[2:]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_single_state(
    ax: plt.Axes,
    t: np.ndarray,
    x_nom: np.ndarray,
    x_proposed: np.ndarray,
    x_baseline: np.ndarray,
    bounds: Tuple[float, float],
    P_history: List[np.ndarray],
    state_idx: int,
    T: int,
    dt: float,
    label: str,
):
    """
    Plot a single state with nominal, proposed, baseline, and funnels.

    Args:
        ax: Matplotlib axes
        t: Time array
        x_nom: Nominal trajectory for this state
        x_proposed: Proposed method trajectory
        x_baseline: Baseline trajectory
        bounds: (lower, upper) state bounds
        P_history: List of P matrices per segment
        state_idx: Which state index (0-3)
        T: Segment length
        dt: Time step
        label: Y-axis label
    """
    N = len(t) - 1

    # Plot state bounds
    ax.axhline(bounds[0], color="green", linestyle="-", linewidth=1, alpha=0.7)
    ax.axhline(bounds[1], color="green", linestyle="-", linewidth=1, alpha=0.7)
    ax.fill_between(t, bounds[0], bounds[1], color="green", alpha=0.05)

    # Plot funnels as shaded regions around nominal
    for seg_idx, P in enumerate(P_history):
        seg_start = seg_idx * T
        seg_end = min((seg_idx + 1) * T, N)
        if seg_start >= N:
            break

        t_seg = t[seg_start:seg_end + 1]
        x_nom_seg = x_nom[seg_start:seg_end + 1]

        # Funnel half-width for this state: sqrt(1/P[i,i])
        if P[state_idx, state_idx] > 0:
            half_width = 1.0 / np.sqrt(P[state_idx, state_idx])
        else:
            half_width = 0.0

        ax.fill_between(
            t_seg,
            x_nom_seg - half_width,
            x_nom_seg + half_width,
            color="gray",
            alpha=0.2,
            label="Funnel" if seg_idx == 0 else None,
        )

    # Plot trajectories
    ax.plot(t, x_nom, "k-", linewidth=2, label="Nominal (twin)")
    ax.plot(t, x_baseline, "gray", linestyle="--", linewidth=1.5, label="Baseline")
    ax.plot(t, x_proposed, "r-", linewidth=1.5, label="Proposed")

    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, t[-1]])


def plot_deviation_comparison(
    results,
    baseline_results,
    dt: float = 0.01,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ||eta|| over time for proposed vs baseline.

    Args:
        results: SimulationResults from proposed method
        baseline_results: BaselineResults from baseline
        dt: Time step
        save_path: Optional save path

    Returns:
        fig: matplotlib Figure
    """
    N = results.x_trajectory.shape[0] - 1
    t = np.arange(N + 1) * dt

    eta_proposed = np.linalg.norm(results.eta_trajectory, axis=1)
    eta_baseline = np.linalg.norm(baseline_results.eta_trajectory, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.semilogy(t, eta_proposed, "r-", linewidth=1.5, label="Proposed")
    ax.semilogy(t, eta_baseline, "gray", linestyle="--", linewidth=1.5, label="Baseline")

    # Mark segment boundaries
    T = SIM_PARAMS["T"]
    for seg in range(1, N // T):
        ax.axvline(seg * T * dt, color="blue", linestyle=":", alpha=0.3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|\eta\|$")
    ax.set_title("Tracking Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_control_inputs(
    results,
    baseline_results,
    dt: float = 0.01,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot control inputs over time.

    Args:
        results: SimulationResults from proposed method
        baseline_results: BaselineResults from baseline
        dt: Time step
        save_path: Optional save path

    Returns:
        fig: matplotlib Figure
    """
    N = results.u_trajectory.shape[0]
    t = np.arange(N) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    labels = [r"$\tau_1$ [Nm]", r"$\tau_2$ [Nm]"]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, results.u_nom[:, i], "k-", linewidth=2, label="Nominal")
        ax.plot(t, results.u_trajectory[:, i], "r-", linewidth=1, alpha=0.7, label="Proposed")
        ax.plot(t, baseline_results.u_trajectory[:, i], "gray", linestyle="--", linewidth=1, label="Baseline")

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
