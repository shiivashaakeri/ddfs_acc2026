"""Sensitivity of final ||η(N)|| to (α, μ)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.base_runner import run_full_algorithm


def main(save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    alphas = [0.90, 0.95, 0.98, 0.99]
    mus = [1.01, 1.05, 1.10, 1.50]
    grid = np.full((len(alphas), len(mus)), np.nan)
    rows = []
    for i, a in enumerate(alphas):
        for j, m in enumerate(mus):
            print(f"α={a}, μ={m}…", end=" ", flush=True)
            try:
                r = run_full_algorithm(alpha=a, mu=m)
                grid[i, j] = r.eta_final
                rows.append((a, m, r.eta_final, r.eta_max, r.total_wall_time))
                print(f"η_final={r.eta_final:.4f}")
            except Exception as e:
                rows.append((a, m, float("nan"), float("nan"), float("nan")))
                print(f"FAIL ({type(e).__name__})")

    csv_path = Path(save_dir) / "sensitivity_alpha_mu.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mu", "eta_final", "eta_max", "wall_time_s"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(np.log10(grid + 1e-12), origin="lower", aspect="auto",
                   extent=[mus[0] - 0.05, mus[-1] + 0.05, alphas[0] - 0.01, alphas[-1] + 0.01])
    ax.set_xticks(mus); ax.set_yticks(alphas)
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"$\log_{10} ||\eta(N)||$")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "sensitivity_alpha_mu.png", dpi=150)
    print(f"saved {csv_path} and sensitivity_alpha_mu.png")


if __name__ == "__main__":
    main()
