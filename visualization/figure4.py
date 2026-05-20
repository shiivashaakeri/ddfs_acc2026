"""Reproduce Fig. 4 of the paper: paired q1, q2 (left) and dq1, dq2 (right).

Shows nominal (black), plant under proposed control (red), plant under
twin-controller baseline (dashed grey), feasibility bounds (green), and
per-segment funnel envelopes (gray fill).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_figure4(
    t: np.ndarray,
    x_nom: np.ndarray,
    x_plant: np.ndarray,
    x_baseline: np.ndarray,
    funnel_envelopes: list[np.ndarray] | None,
    funnel_segments: list[tuple[int, int]] | None,
    state_bounds: list[tuple[float, float]],
    funnel_envelopes_raw: list[np.ndarray] | None = None,
    save_path: str | None = None,
):
    """4-panel plot (q1, q2, dq1, dq2).

    funnel_envelopes : per-segment shape matrices used for the gray-fill band
        (these are the inflated funnels that contain the plant trajectory).
    funnel_envelopes_raw : optional per-segment SDP-certified funnels before
        inflation; rendered as a thin dashed border on each segment so the
        plant-vs-LS gap is visible.
    """
    labels = [r"$q_1$ (rad)", r"$q_2$ (rad)", r"$\dot{q}_1$ (rad/s)", r"$\dot{q}_2$ (rad/s)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes = axes.ravel()

    for i in range(4):
        ax = axes[i]
        if funnel_envelopes is not None and funnel_segments is not None:
            lower = np.full(len(t), np.nan)
            upper = np.full(len(t), np.nan)
            for (P_shape, (k_lo, k_hi)) in zip(funnel_envelopes, funnel_segments):
                semi = float(np.sqrt(max(P_shape[i, i], 0.0)))
                for k in range(k_lo, min(k_hi, len(t))):
                    lower[k] = x_nom[k, i] - semi
                    upper[k] = x_nom[k, i] + semi
            ax.fill_between(t, lower, upper, color="gray", alpha=0.25, label="funnel (containing plant)")
        if funnel_envelopes_raw is not None and funnel_segments is not None:
            lower_r = np.full(len(t), np.nan)
            upper_r = np.full(len(t), np.nan)
            for (P_shape, (k_lo, k_hi)) in zip(funnel_envelopes_raw, funnel_segments):
                semi = float(np.sqrt(max(P_shape[i, i], 0.0)))
                for k in range(k_lo, min(k_hi, len(t))):
                    lower_r[k] = x_nom[k, i] - semi
                    upper_r[k] = x_nom[k, i] + semi
            ax.plot(t, lower_r, color="dimgray", ls="--", lw=0.7, label="SDP-certified (for LS estimate)")
            ax.plot(t, upper_r, color="dimgray", ls="--", lw=0.7)
        # Feasibility bounds
        lo, hi = state_bounds[i]
        ax.axhline(lo, color="g", ls=":", lw=0.8)
        ax.axhline(hi, color="g", ls=":", lw=0.8)
        # Trajectories
        ax.plot(t, x_baseline[:, i], color="gray", ls="--", lw=1.0, label="twin ctrl on plant (baseline)")
        ax.plot(t, x_nom[:, i], "k-", lw=1.5, label="nominal (twin)")
        ax.plot(t, x_plant[:, i], "r-", lw=1.3, label="online control (plant)")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("time (s)")
    axes[3].set_xlabel("time (s)")
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axes
