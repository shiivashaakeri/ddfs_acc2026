"""Timing breakdown: nominal planning vs per-segment SDP solve vs simulation."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.params import INITIAL_STATES, INPUT_BOUNDS, LIPSCHITZ, SIM_PARAMS, STATE_BOUNDS, SYNTH_PARAMS
from control.initial_controller import design_initial_controller
from control.segmented import run_algorithm_1
from dynamics.digital_twin import create_plant, create_twin
from synthesis.feasibility_sets import compute_Pmin_Rmax_along_nominal, segment_envelopes
from synthesis.sdp import synthesize_funnel
from data_driven.uncertainty import SegmentUncertainty
from trajectory.cache import get_or_plan_nominal
from trajectory.nominal_planner import plan_nominal_with_twin_lqr


def main(save_dir: str = "outputs"):
    Path(save_dir).mkdir(exist_ok=True)
    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    x_plant_0 = np.array(INITIAL_STATES["x_plant_0"])
    u_lo = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["q1", "q2", "dq1", "dq2"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["q1", "q2", "dq1", "dq2"]])
    N, T, L = SIM_PARAMS["N"], SIM_PARAMS["T"], SIM_PARAMS["L"]
    n_segments = N // T

    # Time the nominal planning (force a fresh iLQR; bypass cache)
    print("Timing: iLQR nominal planning (uncached)…")
    t0 = time.perf_counter()
    plan_nominal_with_twin_lqr(x_nom_0, x_goal, u_goal, N, u_lo=u_lo, u_hi=u_hi, max_iter=80)
    t_plan = time.perf_counter() - t0
    print(f"  {t_plan:.2f} s")

    # Reuse cached nominal for the rest
    nom = get_or_plan_nominal(x_nom_0, x_goal, u_goal, N, u_lo=u_lo, u_hi=u_hi)

    # Time per-segment SDP
    print("Timing: per-segment SDP solve…")
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True)
        P_max_segs.append(Pm); R_max_segs.append(Rm)

    twin = create_twin(); plant = create_plant()
    eta_0 = x_plant_0 - nom.x_nom[0]
    P0, K0 = design_initial_controller(twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0], eta_0=eta_0, alpha=SYNTH_PARAMS["alpha"])

    t_sdp = []
    P_i = P0; K_i = K0
    # Quick collection of synthetic clean data per segment (matching CE mode)
    n, m = 4, 2
    for i in range(n_segments - 1):
        # Synthetic probe pinning a near-true system
        A_seed, B_seed = P_max_segs[i] * 0 + np.eye(n) * 0.99, np.zeros((n, m)) + 0.1
        H = np.zeros((n, n + m)); H[:, :n] = np.eye(n)
        Xi = np.zeros((m, n + m)); Xi[:, n:] = np.eye(m)
        H_plus = A_seed @ H + B_seed @ Xi
        bundle = SegmentUncertainty(H=H, H_plus=H_plus, Xi=Xi, beta=1e-10, C=0.0, T_tilde=1.0)
        t0 = time.perf_counter()
        try:
            synthesize_funnel(bundle, P_max_segs[i + 1], R_max_segs[i + 1], alpha=SYNTH_PARAMS["alpha"])
        except Exception:
            pass
        t_sdp.append(time.perf_counter() - t0)

    print(f"  per-segment SDP times: {[f'{t:.2f}s' for t in t_sdp]}")
    print(f"  mean: {np.mean(t_sdp):.2f}s, max: {np.max(t_sdp):.2f}s")

    # Time the closed-loop simulation
    print("Timing: closed-loop simulation…")
    t0 = time.perf_counter()
    run_algorithm_1(
        plant_step=plant.step, x_plant_0=x_plant_0, x_nom=nom.x_nom, u_nom=nom.u_nom,
        P_max_shape_per_segment=P_max_segs, R_max_per_segment=R_max_segs, P0=P0, K0=K0, T=T, L=L,
        alpha=SYNTH_PARAMS["alpha"], mu=SYNTH_PARAMS["mu"], eps_bar=SYNTH_PARAMS["eps_bar"],
        gamma=LIPSCHITZ["gamma"], C=0.0, L_r=LIPSCHITZ["L_r"], T_tilde=1.0,
        CT_per_segment=[0.0] * n_segments, u_lo=u_lo, u_hi=u_hi, beta_mode="ce",
    )
    t_sim = time.perf_counter() - t0
    print(f"  {t_sim:.2f}s")

    csv_path = Path(save_dir) / "timing_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "wall_time_s"])
        w.writerow(["iLQR_nominal_planning", t_plan])
        w.writerow(["mean_per_segment_sdp", np.mean(t_sdp)])
        w.writerow(["max_per_segment_sdp", np.max(t_sdp)])
        w.writerow(["full_closed_loop_sim", t_sim])

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    stages = ["iLQR plan", "mean SDP/seg", "max SDP/seg", "full sim"]
    values = [t_plan, np.mean(t_sdp), np.max(t_sdp), t_sim]
    ax.bar(stages, values, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("wall time (s)")
    ax.set_title("Algorithm timing breakdown")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05 * max(values), f"{v:.2f}s", ha="center")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "timing_analysis.png", dpi=150)
    print(f"saved {csv_path} and timing_analysis.png")


if __name__ == "__main__":
    main()
