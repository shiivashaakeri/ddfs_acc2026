"""Per-component plots of states and inputs over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_states_inputs(
    t: np.ndarray,
    x_nom: np.ndarray,
    x_plant: np.ndarray,
    u_nom: np.ndarray,
    u_plant: np.ndarray,
    state_bounds: list[tuple[float, float]],
    input_bounds: list[tuple[float, float]],
    save_path: str | None = None,
):
    n = x_nom.shape[1]
    m = u_nom.shape[1]
    fig, axes = plt.subplots(n + m, 1, figsize=(9, 1.6 * (n + m)), sharex=True)
    labels_x = [r"$q_1$", r"$q_2$", r"$\dot{q}_1$", r"$\dot{q}_2$"]
    labels_u = [r"$\tau_1$", r"$\tau_2$"]
    for i in range(n):
        ax = axes[i]
        ax.plot(t, x_nom[:, i], "k-", lw=1.5, label="nominal (twin)" if i == 0 else None)
        ax.plot(t, x_plant[:, i], "r-", lw=1.2, label="online control (plant)" if i == 0 else None)
        ax.axhline(state_bounds[i][0], color="g", ls=":", lw=0.8)
        ax.axhline(state_bounds[i][1], color="g", ls=":", lw=0.8)
        ax.set_ylabel(labels_x[i])
        ax.grid(True, alpha=0.3)
    for j in range(m):
        ax = axes[n + j]
        ax.plot(t[:-1], u_nom[:, j], "k-", lw=1.5)
        ax.plot(t[:-1], u_plant[:, j], "r-", lw=1.2)
        ax.axhline(input_bounds[j][0], color="g", ls=":", lw=0.8)
        ax.axhline(input_bounds[j][1], color="g", ls=":", lw=0.8)
        ax.set_ylabel(labels_u[j])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, axes
