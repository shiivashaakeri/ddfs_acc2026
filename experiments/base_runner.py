"""Shared scaffolding for reviewer experiments.

A single function `run_full_algorithm` that wraps the setup → plan → synthesize
→ simulate pipeline so each experiment can override one parameter and measure
the result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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
from control.segmented import AlgorithmOutputs, run_algorithm_1
from dynamics.digital_twin import create_plant, create_twin
from simulation.baseline import run_twin_only_baseline
from synthesis.feasibility_sets import compute_Pmin_Rmax_along_nominal, segment_envelopes
from trajectory.cache import get_or_plan_nominal


@dataclass
class RunResult:
    outputs: AlgorithmOutputs
    x_baseline: np.ndarray
    eta_final: float
    eta_max: float
    sdp_solve_time_per_segment: list[float]
    total_wall_time: float


def run_full_algorithm(
    T: int | None = None,
    L: int | None = None,
    alpha: float | None = None,
    mu: float | None = None,
    eps_bar: float | None = None,
    mismatch_scale: float = 1.0,
    beta_mode: str = "ce",
    seed: int = 0,
    verbose: bool = False,
) -> RunResult:
    """Run the full pipeline. Any None argument is filled from config defaults.

    mismatch_scale: if != 1.0, rescales the twin parameters' offset from plant
    (1.0 = paper default; 0.0 = identical twin/plant; 2.0 = double mismatch).
    """
    T = T if T is not None else SIM_PARAMS["T"]
    L = L if L is not None else SIM_PARAMS["L"]
    alpha = alpha if alpha is not None else SYNTH_PARAMS["alpha"]
    mu = mu if mu is not None else SYNTH_PARAMS["mu"]
    eps_bar = eps_bar if eps_bar is not None else SYNTH_PARAMS["eps_bar"]

    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    x_plant_0 = np.array(INITIAL_STATES["x_plant_0"])
    u_lo = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])
    x_lo = np.array([STATE_BOUNDS[k][0] for k in ["q1", "q2", "dq1", "dq2"]])
    x_hi = np.array([STATE_BOUNDS[k][1] for k in ["q1", "q2", "dq1", "dq2"]])
    N = SIM_PARAMS["N"]
    n_segments = N // T

    # ---- Mismatch scaling (rescale twin parameters)
    if mismatch_scale != 1.0:
        from config.params import PLANT_PARAMS, TWIN_PARAMS
        scaled = {}
        for k in PLANT_PARAMS:
            if k == "g":
                scaled[k] = PLANT_PARAMS[k]
            else:
                scaled[k] = PLANT_PARAMS[k] + mismatch_scale * (TWIN_PARAMS[k] - PLANT_PARAMS[k])
        from dynamics.robot_arm import RobotArm2DoF
        from dynamics.discretization import DiscreteSystem
        twin_arm = RobotArm2DoF(**{k: v for k, v in scaled.items() if k != "g"}, g=scaled["g"])
        twin = DiscreteSystem(twin_arm.dynamics, SIM_PARAMS["dt"])
    else:
        twin = create_twin()
    plant = create_plant()

    nom = get_or_plan_nominal(x_nom_0, x_goal, u_goal, N, u_lo=u_lo, u_hi=u_hi)
    P_env_seq, R_env_seq = compute_Pmin_Rmax_along_nominal(
        nom.x_nom, nom.u_nom, x_lo, x_hi, u_lo, u_hi, shape_convention=True
    )
    P_max_segs, R_max_segs = [], []
    for i in range(n_segments):
        Pm, Rm = segment_envelopes(P_env_seq, R_env_seq, range(i * T, (i + 1) * T), shape_convention=True)
        P_max_segs.append(Pm)
        R_max_segs.append(Rm)

    eta_0 = x_plant_0 - nom.x_nom[0]
    P0, K0 = design_initial_controller(
        twin, nom.x_nom[0], nom.u_nom[0], P_max_segs[0], eta_0=eta_0, alpha=alpha
    )

    t_start = time.perf_counter()
    outputs = run_algorithm_1(
        plant_step=plant.step, x_plant_0=x_plant_0,
        x_nom=nom.x_nom, u_nom=nom.u_nom,
        P_max_shape_per_segment=P_max_segs, R_max_per_segment=R_max_segs,
        P0=P0, K0=K0, T=T, L=L,
        alpha=alpha, mu=mu, eps_bar=eps_bar,
        gamma=LIPSCHITZ["gamma"], C=0.0, L_r=LIPSCHITZ["L_r"], T_tilde=1.0,
        CT_per_segment=[0.0] * n_segments,
        u_lo=u_lo, u_hi=u_hi,
        beta_mode=beta_mode, rng_seed=seed, verbose=verbose,
    )
    wall_time = time.perf_counter() - t_start

    x_baseline = run_twin_only_baseline(plant.step, x_plant_0, nom.u_nom)

    eta_final = float(np.linalg.norm(outputs.x_plant_full[-1] - nom.x_nom[-1]))
    eta_max = float(np.max([np.linalg.norm(s.eta, axis=1).max() for s in outputs.segments]))
    # Per-segment SDP solve time: not measured separately here; assign per-segment fraction
    if outputs.sdp_designs:
        per_sdp = wall_time / max(len(outputs.sdp_designs), 1)
        per_seg = [per_sdp] * len(outputs.sdp_designs)
    else:
        per_seg = []

    return RunResult(
        outputs=outputs,
        x_baseline=x_baseline,
        eta_final=eta_final,
        eta_max=eta_max,
        sdp_solve_time_per_segment=per_seg,
        total_wall_time=wall_time,
    )
