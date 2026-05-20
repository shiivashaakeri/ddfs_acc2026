"""Cart-pole case study (stabilisation by default; swing-up available)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from control.initial_controller import design_initial_controller
from control.segmented import run_algorithm_1
from linearization.jacobians import jacobians_discrete
from linearization.lipschitz import estimate_jacobian_variation_bound
from simulation.baseline import run_twin_only_baseline
from synthesis.feasibility_sets import (
    compute_Pmin_Rmax_along_nominal,
    segment_envelopes,
)
from trajectory.nominal_planner import NominalTrajectory

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


def _lqr_rollout_nominal(twin, x_nom_0, x_goal, u_goal, N, u_lo, u_hi):
    """LQR-rollout planner: linearise at the goal, design DLQR, simulate."""
    from scipy.linalg import solve_discrete_are
    A, B = jacobians_discrete(twin.step, x_goal, u_goal)
    Q = np.diag([10.0, 1.0, 30.0, 1.0])
    R = np.array([[1.0]])
    P_ric = solve_discrete_are(A, B, Q, R)
    K_lqr = np.linalg.solve(R + B.T @ P_ric @ B, B.T @ P_ric @ A)
    n, m = A.shape[0], B.shape[1]
    x_nom = np.zeros((N + 1, n)); x_nom[0] = x_nom_0
    u_nom = np.zeros((N, m))
    for k in range(N):
        u_k = u_goal - K_lqr @ (x_nom[k] - x_goal)
        u_k = np.clip(u_k, u_lo, u_hi)
        u_nom[k] = u_k
        x_nom[k + 1] = twin.step(x_nom[k], u_k)
    return NominalTrajectory(x_nom=x_nom, u_nom=u_nom, x_goal=x_goal, u_goal=u_goal)


def main(save_dir: str = "outputs", beta_mode: str = "ls_lqr"):
    os.makedirs(save_dir, exist_ok=True)

    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    x_plant_0 = np.array(INITIAL_STATES["x_plant_0"])
    u_lo = np.array([INPUT_BOUNDS["force"][0]])
    u_hi = np.array([INPUT_BOUNDS["force"][1]])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["p", "p_dot", "theta", "theta_dot"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["p", "p_dot", "theta", "theta_dot"]])
    N, T, L = SIM_PARAMS["N"], SIM_PARAMS["T"], SIM_PARAMS["L"]
    dt = SIM_PARAMS["dt"]
    n_segments = N // T

    print(f"[1/4] Planning nominal trajectory (LQR rollout)…  beta_mode={beta_mode}")
    twin_for_planning = create_twin()
    nom = _lqr_rollout_nominal(twin_for_planning, x_nom_0, x_goal, u_goal, N, u_lo, u_hi)
    print(f"    terminal err: {np.linalg.norm(nom.x_nom[-1] - x_goal):.4f}")
    print(f"    increment bound v: {nom.increment_bound():.4f}")
    print(f"    state ranges: p∈[{nom.x_nom[:,0].min():.2f},{nom.x_nom[:,0].max():.2f}], "
          f"θ∈[{nom.x_nom[:,2].min():.2f},{nom.x_nom[:,2].max():.2f}]")

    print("[2/4] Feasibility envelopes + initial controller…")
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True)
        P_max_segs.append(Pm); R_max_segs.append(Rm)

    twin = create_twin(); plant = create_plant()
    eta_0 = x_plant_0 - nom.x_nom[0]
    print(f"    initial deviation ||η_0|| = {np.linalg.norm(eta_0):.4f}")
    # Strong position weight to suppress non-normal transient growth from the
    # cart-pole's integrator pair (eigvals near 1). Without this the linear
    # closed-loop transient amplifies before settling and the nonlinear plant
    # escapes the linear regime.
    P0, K0 = design_initial_controller(
        twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0],
        eta_0=eta_0, alpha=SYNTH_PARAMS["alpha"],
        Q=np.diag([100.0, 5.0, 50.0, 2.0]), R=np.array([[1.0]]),
    )
    print(f"    K0 = {K0[0]}")

    CT_per_segment = []
    for i in range(n_segments):
        CT = estimate_jacobian_variation_bound(
            plant.step, nom.x_nom, nom.u_nom, i * T, min((i + 2) * T, N), safety=2.0
        )
        CT_per_segment.append(CT)
    print(f"    empirical C·T̃: {[f'{c:.4f}' for c in CT_per_segment]}")

    print(f"[3/4] Running Algorithm 1 ({beta_mode=})…")
    outputs = run_algorithm_1(
        plant_step=plant.step, x_plant_0=x_plant_0,
        x_nom=nom.x_nom, u_nom=nom.u_nom,
        P_max_shape_per_segment=P_max_segs, R_max_per_segment=R_max_segs,
        P0=P0, K0=K0, T=T, L=L,
        alpha=SYNTH_PARAMS["alpha"], mu=SYNTH_PARAMS["mu"], eps_bar=SYNTH_PARAMS["eps_bar"],
        gamma=LIPSCHITZ["gamma"], C=LIPSCHITZ["C"], L_r=LIPSCHITZ["L_r"],
        T_tilde=LIPSCHITZ["T_tilde"], CT_per_segment=CT_per_segment,
        u_lo=u_lo, u_hi=u_hi, beta_mode=beta_mode, verbose=True,
        ls_lqr_Q=np.diag([100.0, 5.0, 50.0, 2.0]),
        ls_lqr_R=np.array([[1.0]]),
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

    state_bounds_list = [STATE_BOUNDS[k] for k in ["p", "p_dot", "theta", "theta_dot"]]
    t = np.arange(N + 1) * dt

    out_path = Path(save_dir) / f"cartpole_figure_{beta_mode}.png"
    _plot_4d_case(
        t, nom.x_nom, outputs.x_plant_full, x_baseline,
        funnel_envelopes, funnel_envelopes_raw, funnel_segments,
        state_bounds_list, out_path
    )
    print(f"    figure: {out_path}")
    return outputs, eta_final


def _plot_4d_case(t, x_nom, x_plant, x_baseline, env_inflated, env_raw, segs, bounds, save_path):
    """2x2 plot for the cart-pole's four states."""
    import matplotlib.pyplot as plt
    labels = [r"$p$ (m)", r"$\dot{p}$ (m/s)", r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    axes = axes.ravel()
    for i in range(4):
        ax = axes[i]
        lo = np.full(len(t), np.nan); hi = np.full(len(t), np.nan)
        for (P, (k_lo, k_hi)) in zip(env_inflated, segs):
            semi = float(np.sqrt(max(P[i, i], 0.0)))
            for k in range(k_lo, min(k_hi, len(t))):
                lo[k] = x_nom[k, i] - semi; hi[k] = x_nom[k, i] + semi
        ax.fill_between(t, lo, hi, color="gray", alpha=0.25, label="funnel (containing plant)")
        lo_r = np.full(len(t), np.nan); hi_r = np.full(len(t), np.nan)
        for (P, (k_lo, k_hi)) in zip(env_raw, segs):
            semi = float(np.sqrt(max(P[i, i], 0.0)))
            for k in range(k_lo, min(k_hi, len(t))):
                lo_r[k] = x_nom[k, i] - semi; hi_r[k] = x_nom[k, i] + semi
        ax.plot(t, lo_r, color="dimgray", ls="--", lw=0.7, label="SDP-certified (LS)")
        ax.plot(t, hi_r, color="dimgray", ls="--", lw=0.7)
        ax.axhline(bounds[i][0], color="g", ls=":", lw=0.8)
        ax.axhline(bounds[i][1], color="g", ls=":", lw=0.8)
        ax.plot(t, x_baseline[:, i], color="gray", ls="--", lw=1.0, label="twin ctrl on plant")
        ax.plot(t, x_nom[:, i], "k-", lw=1.5, label="nominal")
        ax.plot(t, x_plant[:, i], "r-", lw=1.3, label="online control")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("time (s)"); axes[3].set_xlabel("time (s)")
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


if __name__ == "__main__":
    main()
