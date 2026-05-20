"""Sensitivity of final ||η(N)|| and timing to segment length T."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.base_runner import run_full_algorithm


def main(save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    T_values = [50, 75, 100, 150, 200]
    rows = []
    for T in T_values:
        # Adjust L so it always fits: L = min(60, T-10)
        L = min(60, max(T - 10, 10))
        print(f"Running T={T}, L={L}…")
        try:
            r = run_full_algorithm(T=T, L=L)
            rows.append((T, L, r.eta_final, r.eta_max, r.total_wall_time))
            print(f"  η_final={r.eta_final:.4f}, η_max={r.eta_max:.4f}, time={r.total_wall_time:.1f}s")
        except Exception as e:
            print(f"  FAIL ({type(e).__name__})")
            rows.append((T, L, float("nan"), float("nan"), float("nan")))

    csv_path = Path(save_dir) / "sensitivity_T.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T", "L", "eta_final", "eta_max", "wall_time_s"])
        w.writerows(rows)

    Ts = [r[0] for r in rows if r[2] == r[2]]  # filter NaN
    etas = [r[2] for r in rows if r[2] == r[2]]
    times = [r[4] for r in rows if r[2] == r[2]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Ts, etas, "o-")
    axes[0].set_xlabel("T (segment length)")
    axes[0].set_ylabel(r"final $||\eta(N)||$")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(Ts, times, "s-", color="C1")
    axes[1].set_xlabel("T (segment length)")
    axes[1].set_ylabel("wall time (s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "sensitivity_T.png", dpi=150)
    print(f"saved {csv_path} and sensitivity_T.png")


if __name__ == "__main__":
    main()
