"""Reproduce Fig. 4 of the paper.

Runs the full segmented controller on the 2-DoF arm and produces the four-panel
tracking + funnel + baseline figure.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from config.params import (
    INITIAL_STATES,
    INPUT_BOUNDS,
    LIPSCHITZ,
    SIM_PARAMS,
    STATE_BOUNDS,
    SYNTH_PARAMS,
)
from control.initial_controller import design_initial_controller
from control.segmented import run_algorithm_1
from dynamics.digital_twin import create_plant, create_twin
from simulation.baseline import run_twin_only_baseline
from synthesis.feasibility_sets import compute_Pmin_Rmax_along_nominal, segment_envelopes
from trajectory.cache import get_or_plan_nominal
from visualization.figure4 import plot_figure4


def main(save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)

    # ---- Setup
    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    x_plant_0 = np.array(INITIAL_STATES["x_plant_0"])
    u_lo = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["q1", "q2", "dq1", "dq2"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["q1", "q2", "dq1", "dq2"]])
    N, T, L = SIM_PARAMS["N"], SIM_PARAMS["T"], SIM_PARAMS["L"]
    dt = SIM_PARAMS["dt"]
    n_segments = N // T

    # ---- Plan nominal on twin
    print("[1/4] Planning nominal trajectory (iLQR on twin)…")
    nom = get_or_plan_nominal(x_nom_0, x_goal, u_goal, N, u_lo=u_lo, u_hi=u_hi, verbose=True)
    print(f"    terminal error: {np.linalg.norm(nom.x_nom[-1] - x_goal):.4f}")

    # ---- Feasibility envelopes (MVIE) along nominal
    print("[2/4] Computing per-segment feasibility envelopes…")
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True)
        P_max_segs.append(Pm)
        R_max_segs.append(Rm)

    # ---- Initial controller K_0 from twin LQR
    twin = create_twin()
    plant = create_plant()
    eta_0 = x_plant_0 - nom.x_nom[0]
    print(f"    initial deviation ||η_0|| = {np.linalg.norm(eta_0):.4f}")
    P0, K0 = design_initial_controller(
        twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0], eta_0=eta_0, alpha=SYNTH_PARAMS["alpha"]
    )

    # ---- Run Algorithm 1
    print("[3/4] Running Algorithm 1 (segmented funnel control)…")
    outputs = run_algorithm_1(
        plant_step=plant.step,
        x_plant_0=x_plant_0,
        x_nom=nom.x_nom,
        u_nom=nom.u_nom,
        P_max_shape_per_segment=P_max_segs,
        R_max_per_segment=R_max_segs,
        P0=P0,
        K0=K0,
        T=T,
        L=L,
        alpha=SYNTH_PARAMS["alpha"],
        mu=SYNTH_PARAMS["mu"],
        eps_bar=SYNTH_PARAMS["eps_bar"],
        gamma=LIPSCHITZ["gamma"],
        C=0.0,
        L_r=LIPSCHITZ["L_r"],
        T_tilde=1.0,
        CT_per_segment=[0.0] * n_segments,
        u_lo=u_lo,
        u_hi=u_hi,
        beta_mode="ce",
        verbose=True,
    )
    print(f"    final ||η(N)|| = {np.linalg.norm(outputs.x_plant_full[-1] - nom.x_nom[-1]):.4f}")
    print(f"    paper reports ||η(N)|| = 4.725e-2")

    # ---- Baseline: twin's nominal controller applied directly to plant
    print("[4/4] Computing baseline (twin controller on plant)…")
    x_baseline = run_twin_only_baseline(plant.step, x_plant_0, nom.u_nom)

    # ---- Plot
    # Use P_0 from the initial-controller design (not the feasibility envelope)
    # so the segment-0 funnel reflects the actual certified region.
    funnel_envelopes = [P0] + [d.P for d in outputs.sdp_designs]
    funnel_segments = [(i * T, (i + 1) * T) for i in range(n_segments)]
    t = np.arange(N + 1) * dt
    state_bounds_list = [STATE_BOUNDS[k] for k in ["q1", "q2", "dq1", "dq2"]]
    out_path = Path(save_dir) / "figure4.png"
    plot_figure4(
        t=t,
        x_nom=nom.x_nom,
        x_plant=outputs.x_plant_full,
        x_baseline=x_baseline,
        funnel_envelopes=funnel_envelopes,
        funnel_segments=funnel_segments,
        state_bounds=state_bounds_list,
        save_path=str(out_path),
    )
    print(f"    figure saved to {out_path}")

    return outputs, x_baseline


if __name__ == "__main__":
    main()
