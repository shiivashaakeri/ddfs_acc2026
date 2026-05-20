"""Reproduce the funnel-control case study on the inverted pendulum."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from control.initial_controller import design_initial_controller
from control.segmented import run_algorithm_1
from linearization.lipschitz import estimate_jacobian_variation_bound
from simulation.baseline import run_twin_only_baseline
from synthesis.feasibility_sets import (
    compute_Pmin_Rmax_along_nominal,
    segment_envelopes,
)
from trajectory.cache import get_or_plan_nominal
from trajectory.nominal_planner import NominalTrajectory
from visualization.figure4 import plot_figure4

from .config import (
    INITIAL_STATES,
    INPUT_BOUNDS,
    LIPSCHITZ,
    SIM_PARAMS,
    STATE_BOUNDS,
    SYNTH_PARAMS,
    create_plant,
    create_twin,
)


def main(save_dir: str = "outputs", beta_mode: str = "ce"):
    """Run the full pipeline on the inverted pendulum.

    beta_mode : "paper" tries the matrix S-lemma's Eq. (16) β; the 2D pendulum
                is small enough that this may be feasible. "ce" uses the
                certainty-equivalence relaxation that's the arm case study's
                default.
    """
    os.makedirs(save_dir, exist_ok=True)

    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    x_plant_0 = np.array(INITIAL_STATES["x_plant_0"])
    u_lo = np.array([INPUT_BOUNDS["tau"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau"][1]])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["theta", "theta_dot"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["theta", "theta_dot"]])
    N, T, L = SIM_PARAMS["N"], SIM_PARAMS["T"], SIM_PARAMS["L"]
    dt = SIM_PARAMS["dt"]
    n_segments = N // T

    print(f"[1/4] Planning nominal trajectory (LQR rollout)…  beta_mode={beta_mode}")
    twin_for_planning = create_twin()
    # Direct LQR-rollout: linearise twin at the upright equilibrium, design
    # DLQR, and roll the closed-loop policy forward from x_nom_0. For the
    # inverted-pendulum stabilisation this gives a clean direct trajectory,
    # unlike free-form iLQR which tends to swing the long way around.
    from linearization.jacobians import jacobians_discrete
    from scipy.linalg import solve_discrete_are
    A_d, B_d = jacobians_discrete(twin_for_planning.step, x_goal, u_goal)
    Q_lqr = np.diag([10.0, 1.0]); R_lqr = np.array([[1.0]])
    P_ric = solve_discrete_are(A_d, B_d, Q_lqr, R_lqr)
    K_nom = np.linalg.solve(R_lqr + B_d.T @ P_ric @ B_d, B_d.T @ P_ric @ A_d)
    x_nom_arr = np.zeros((N + 1, 2))
    u_nom_arr = np.zeros((N, 1))
    x_nom_arr[0] = x_nom_0
    for k in range(N):
        u_k = u_goal - K_nom @ (x_nom_arr[k] - x_goal)
        u_k = np.clip(u_k, u_lo, u_hi)
        u_nom_arr[k] = u_k
        x_nom_arr[k + 1] = twin_for_planning.step(x_nom_arr[k], u_k)
    nom = NominalTrajectory(x_nom=x_nom_arr, u_nom=u_nom_arr, x_goal=x_goal, u_goal=u_goal)
    print(f"    terminal err: {np.linalg.norm(nom.x_nom[-1] - x_goal):.4f}")
    print(f"    increment bound v: {nom.increment_bound():.4f}")
    print(f"    theta range: [{nom.x_nom[:, 0].min():.3f}, {nom.x_nom[:, 0].max():.3f}]")
    print(f"    tau range: [{nom.u_nom[:, 0].min():.3f}, {nom.u_nom[:, 0].max():.3f}]")
    print(f"    terminal err: {np.linalg.norm(nom.x_nom[-1] - x_goal):.4f}")
    print(f"    increment bound v: {nom.increment_bound():.4f}")

    print("[2/4] Computing per-segment feasibility envelopes…")
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(
            P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True
        )
        P_max_segs.append(Pm)
        R_max_segs.append(Rm)

    twin = create_twin()
    plant = create_plant()
    eta_0 = x_plant_0 - nom.x_nom[0]
    print(f"    initial deviation ||η_0|| = {np.linalg.norm(eta_0):.4f}")
    P0, K0 = design_initial_controller(
        twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0],
        eta_0=eta_0, alpha=SYNTH_PARAMS["alpha"],
        Q=np.diag([1.0, 0.1]), R=np.array([[1.0]]),
    )
    print(f"    K0 = {K0[0]}")

    # Empirical C·T̃ per segment, used by both 'paper' and 'ce' (CT controls Σ^Δ).
    CT_per_segment = []
    for i in range(n_segments):
        k_lo = i * T
        k_hi = min((i + 2) * T, N)
        CT = estimate_jacobian_variation_bound(
            plant.step, nom.x_nom, nom.u_nom, k_lo, k_hi, safety=2.0
        )
        CT_per_segment.append(CT)
    print(f"    empirical C·T̃ per segment: {[f'{c:.4f}' for c in CT_per_segment]}")

    print(f"[3/4] Running Algorithm 1 ({beta_mode=})…")
    outputs = run_algorithm_1(
        plant_step=plant.step, x_plant_0=x_plant_0,
        x_nom=nom.x_nom, u_nom=nom.u_nom,
        P_max_shape_per_segment=P_max_segs, R_max_per_segment=R_max_segs,
        P0=P0, K0=K0, T=T, L=L,
        alpha=SYNTH_PARAMS["alpha"], mu=SYNTH_PARAMS["mu"], eps_bar=SYNTH_PARAMS["eps_bar"],
        gamma=LIPSCHITZ["gamma"], C=LIPSCHITZ["C"], L_r=LIPSCHITZ["L_r"],
        T_tilde=LIPSCHITZ["T_tilde"],
        CT_per_segment=CT_per_segment,
        u_lo=u_lo, u_hi=u_hi,
        beta_mode=beta_mode,
        verbose=True,
    )
    eta_final = np.linalg.norm(outputs.x_plant_full[-1] - nom.x_nom[-1])
    print(f"    final ||η(N)|| = {eta_final:.4f}")

    print("[4/4] Baseline + plot…")
    x_baseline = run_twin_only_baseline(plant.step, x_plant_0, nom.u_nom)

    funnel_envelopes_raw = [P0] + [d.P for d in outputs.sdp_designs]
    funnel_segments = [(i * T, (i + 1) * T) for i in range(n_segments)]

    funnel_envelopes = []
    inflation_factors = []
    for i, P_raw in enumerate(funnel_envelopes_raw):
        if i < len(outputs.segments):
            eta_seg = outputs.segments[i].eta
            P_inv = np.linalg.inv(P_raw)
            quad = np.einsum("ki,ij,kj->k", eta_seg, P_inv, eta_seg)
            c = max(1.0, float(quad.max()))
        else:
            c = 1.0
        funnel_envelopes.append(c * P_raw)
        inflation_factors.append(c)
    print(f"    inflation factors: {[f'{c:.2f}' for c in inflation_factors]}")

    # Pad bounds to match the figure4 signature (q1/q2/dq1/dq2 → θ/θ̇ only here).
    state_bounds_list = [STATE_BOUNDS["theta"], STATE_BOUNDS["theta_dot"]]
    t = np.arange(N + 1) * dt

    # For the pendulum we only have 2 states, so adapt the figure layout.
    out_path = Path(save_dir) / f"pendulum_figure_{beta_mode}.png"
    _plot_2d_case(
        t, nom.x_nom, outputs.x_plant_full, x_baseline,
        funnel_envelopes, funnel_envelopes_raw, funnel_segments,
        state_bounds_list, out_path
    )
    print(f"    figure: {out_path}")
    return outputs, eta_final


def _plot_2d_case(t, x_nom, x_plant, x_baseline, env_inflated, env_raw, segs, bounds, save_path):
    """Two-panel plot (θ, θ̇) — adapter for the 2D pendulum case."""
    import matplotlib.pyplot as plt
    labels = [r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i in range(2):
        ax = axes[i]
        # inflated funnel (gray fill)
        lo = np.full(len(t), np.nan); hi = np.full(len(t), np.nan)
        for (P, (k_lo, k_hi)) in zip(env_inflated, segs):
            semi = float(np.sqrt(max(P[i, i], 0.0)))
            for k in range(k_lo, min(k_hi, len(t))):
                lo[k] = x_nom[k, i] - semi; hi[k] = x_nom[k, i] + semi
        ax.fill_between(t, lo, hi, color="gray", alpha=0.25, label="funnel (containing plant)")
        # raw SDP funnel (dashed)
        lo_r = np.full(len(t), np.nan); hi_r = np.full(len(t), np.nan)
        for (P, (k_lo, k_hi)) in zip(env_raw, segs):
            semi = float(np.sqrt(max(P[i, i], 0.0)))
            for k in range(k_lo, min(k_hi, len(t))):
                lo_r[k] = x_nom[k, i] - semi; hi_r[k] = x_nom[k, i] + semi
        ax.plot(t, lo_r, color="dimgray", ls="--", lw=0.7, label="SDP-certified (LS)")
        ax.plot(t, hi_r, color="dimgray", ls="--", lw=0.7)
        # bounds
        ax.axhline(bounds[i][0], color="g", ls=":", lw=0.8)
        ax.axhline(bounds[i][1], color="g", ls=":", lw=0.8)
        ax.plot(t, x_baseline[:, i], color="gray", ls="--", lw=1.0, label="twin ctrl on plant")
        ax.plot(t, x_nom[:, i], "k-", lw=1.5, label="nominal")
        ax.plot(t, x_plant[:, i], "r-", lw=1.3, label="online control")
        ax.set_xlabel("time (s)"); ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    main()
