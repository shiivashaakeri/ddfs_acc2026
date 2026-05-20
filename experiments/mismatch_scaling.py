"""Effect of plant-twin mismatch magnitude on tracking performance."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.base_runner import run_full_algorithm


def main(save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # 0 = identical twin/plant; 1.0 = paper
    rows = []
    for s in scales:
        print(f"mismatch scale = {s}")
        try:
            r = run_full_algorithm(mismatch_scale=s)
            rows.append((s, r.eta_final, r.eta_max))
            print(f"  η_final={r.eta_final:.4f}, η_max={r.eta_max:.4f}")
        except Exception as e:
            rows.append((s, float("nan"), float("nan")))
            print(f"  FAIL ({type(e).__name__})")

    csv_path = Path(save_dir) / "mismatch_scaling.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mismatch_scale", "eta_final", "eta_max"])
        w.writerows(rows)

    scales_ok = [r[0] for r in rows if r[1] == r[1]]
    etas = [r[1] for r in rows if r[1] == r[1]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scales_ok, etas, "o-")
    ax.set_xlabel("plant–twin mismatch scale (1.0 = paper)")
    ax.set_ylabel(r"final $||\eta(N)||$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "mismatch_scaling.png", dpi=150)
    print(f"saved {csv_path} and mismatch_scaling.png")


if __name__ == "__main__":
    main()
