"""Cart-pole swing-up case study.

Plans the nominal via iLQR (with a sinusoidal initial guess to escape the
stable hanging equilibrium), then runs the segmented funnel controller in
`ls_lqr` mode.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from control.initial_controller import design_initial_controller
from control.segmented import run_algorithm_1
from dynamics.discretization import DiscreteSystem
from linearization.lipschitz import estimate_jacobian_variation_bound
from simulation.baseline import run_twin_only_baseline
from synthesis.feasibility_sets import (
    compute_Pmin_Rmax_along_nominal,
    segment_envelopes,
)
from trajectory.cache import get_or_plan_nominal
from trajectory.nominal_planner import NominalTrajectory, ilqr

from .config import (
    LIPSCHITZ,
    SIM_PARAMS,
    SYNTH_PARAMS,
    TASK_SWING_UP,
    create_plant,
    create_twin,
)
from .reproduce import _plot_4d_case


def _swing_up_nominal(twin, x0, x_goal, u_goal, N, u_lo, u_hi, verbose=False):
    """iLQR from x0 with a sinusoidal initial input to break out of the
    stable hanging equilibrium. Returns a smooth swing-up trajectory."""
    rng_pattern = np.sin(np.arange(N) / 30.0)
    u_init = 5.0 * rng_pattern * np.where(np.arange(N) < N // 3, 1.0, 0.0)
    u_init = u_init[:, None]
    Q = np.diag([5.0, 0.5, 50.0, 5.0])
    R = np.array([[1.0]])
    Qf = 100.0 * Q
    x_seq, u_seq = ilqr(
        twin.step, x0, u_init, x_goal, u_goal, Q, R, Qf,
        u_lo=u_lo, u_hi=u_hi, max_iter=120, verbose=verbose,
    )
    return NominalTrajectory(x_nom=x_seq, u_nom=u_seq, x_goal=x_goal, u_goal=u_goal)


def main(save_dir: str = "outputs", beta_mode: str = "ls_lqr"):
    os.makedirs(save_dir, exist_ok=True)

    x_nom_0 = np.array(TASK_SWING_UP["x_nom_0"])
    x_goal = np.array(TASK_SWING_UP["x_goal"])
    u_goal = np.array(TASK_SWING_UP["u_goal"])
    x_plant_0 = np.array(TASK_SWING_UP["x_plant_0"])
    STATE_BOUNDS = TASK_SWING_UP["STATE_BOUNDS"]
    u_lo = np.array([-25.0]); u_hi = np.array([25.0])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["p", "p_dot", "theta", "theta_dot"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["p", "p_dot", "theta", "theta_dot"]])
    N, T, L = SIM_PARAMS["N"], SIM_PARAMS["T"], SIM_PARAMS["L"]
    dt = SIM_PARAMS["dt"]
    n_segments = N // T

    print(f"[1/4] Planning swing-up trajectory (iLQR)…  beta_mode={beta_mode}")
    twin = create_twin()
    nom = get_or_plan_nominal(
        x_nom_0, x_goal, u_goal, N,
        u_lo=u_lo, u_hi=u_hi, verbose=True,
        system_id="cartpole_swingup",
    )
    # Cache key only includes x0/x_goal/u_goal/N; the default planner inside
    # get_or_plan_nominal is iLQR with default Q/R. For swing-up we need a
    # specific sinusoidal u_init, so we replan if the cache miss happens.
    if abs(nom.x_nom[-1, 2]) > 0.1:
        print("    cache hit gave a bad nominal (not catching at upright); replanning…")
        nom = _swing_up_nominal(twin, x_nom_0, x_goal, u_goal, N, u_lo, u_hi)
        # Overwrite cache so subsequent runs are fast.
        import pickle
        from trajectory.cache import _CACHE_DIR, _key
        key = _key(x_nom_0, x_goal, u_goal, N, "cartpole_swingup")
        with open(_CACHE_DIR / f"nominal_cartpole_swingup_{key}.pkl", "wb") as f:
            pickle.dump(nom, f)
    print(f"    terminal err: {np.linalg.norm(nom.x_nom[-1] - x_goal):.4f}")
    print(f"    swing-up reached θ≈0 by k={int(np.argmax(np.abs(nom.x_nom[:, 2]) < 0.5))}")

    print("[2/4] Feasibility envelopes + initial controller…")
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True)
        P_max_segs.append(Pm); R_max_segs.append(Rm)

    plant = create_plant()
    eta_0 = x_plant_0 - nom.x_nom[0]
    print(f"    initial deviation ||η_0|| = {np.linalg.norm(eta_0):.4f}")
    P0, K0 = design_initial_controller(
        twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0],
        eta_0=eta_0, alpha=SYNTH_PARAMS["alpha"],
        Q=np.diag([10.0, 1.0, 5.0, 1.0]), R=np.array([[1.0]]),
    )
    print(f"    K0 = {K0[0]}")

    CT_per_segment = [
        estimate_jacobian_variation_bound(
            plant.step, nom.x_nom, nom.u_nom, i * T, min((i + 2) * T, N), safety=2.0
        )
        for i in range(n_segments)
    ]
    print(f"    empirical C·T̃: {[f'{c:.3f}' for c in CT_per_segment]}")

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
        ls_lqr_Q=np.diag([10.0, 1.0, 50.0, 5.0]),
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
    print(f"    inflation factors: {[f'{c:.1f}' for c in inflation_factors]}")

    state_bounds_list = [STATE_BOUNDS[k] for k in ["p", "p_dot", "theta", "theta_dot"]]
    t = np.arange(N + 1) * dt
    out_path = Path(save_dir) / f"cartpole_swingup_{beta_mode}.png"
    _plot_4d_case(
        t, nom.x_nom, outputs.x_plant_full, x_baseline,
        funnel_envelopes, funnel_envelopes_raw, funnel_segments,
        state_bounds_list, out_path
    )
    print(f"    figure: {out_path}")
    return outputs, eta_final


if __name__ == "__main__":
    main()
