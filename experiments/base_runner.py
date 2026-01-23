"""
Base experiment runner for DDFS reviewer experiments.

Provides common infrastructure for running experiments and collecting metrics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.params import INITIAL_STATES, LIPSCHITZ, SIM_PARAMS, SYNTH_PARAMS, TWIN_PARAMS, STATE_BOUNDS, INPUT_BOUNDS
from dynamics.digital_twin import create_plant, create_twin
from dynamics.robot_arm import RobotArm2DoF
from trajectory.nominal_planner import generate_nominal_trajectory
from control.segmented_controller import SegmentedFunnelController, design_initial_controller
from simulation.closed_loop import run_simulation
from simulation.baseline import run_baseline, design_twin_controller


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""

    terminal_deviation: float = 0.0
    max_deviation: float = 0.0
    rms_error: float = 0.0
    constraint_violations: int = 0
    violation_magnitude: float = 0.0
    sdp_feasibility_rate: float = 1.0
    sdp_solve_times: List[float] = field(default_factory=list)
    in_funnel_fraction: float = 0.0
    num_segments: int = 0
    success: bool = True
    error_message: str = ""

    # Trajectory data (optional, for detailed analysis)
    eta_trajectory: Optional[np.ndarray] = None
    x_trajectory: Optional[np.ndarray] = None
    u_trajectory: Optional[np.ndarray] = None


def count_constraint_violations(x_trajectory: np.ndarray) -> Tuple[int, float]:
    """
    Count state constraint violations.

    Returns:
        count: number of timesteps with violations
        max_magnitude: maximum violation magnitude
    """
    violations = 0
    max_violation = 0.0

    bounds = [
        (STATE_BOUNDS['q1'][0], STATE_BOUNDS['q1'][1]),
        (STATE_BOUNDS['q2'][0], STATE_BOUNDS['q2'][1]),
        (STATE_BOUNDS['dq1'][0], STATE_BOUNDS['dq1'][1]),
        (STATE_BOUNDS['dq2'][0], STATE_BOUNDS['dq2'][1]),
    ]

    for k in range(len(x_trajectory)):
        x = x_trajectory[k]
        for i, (lb, ub) in enumerate(bounds):
            if x[i] < lb:
                violations += 1
                max_violation = max(max_violation, lb - x[i])
            elif x[i] > ub:
                violations += 1
                max_violation = max(max_violation, x[i] - ub)

    return violations, max_violation


def run_single_experiment(
    T: int = 100,
    L: int = 60,
    alpha: float = 0.98,
    mu: float = 1.02,
    gamma: float = None,
    L_J: float = None,
    L_r: float = None,
    C: float = None,
    x0_plant: np.ndarray = None,
    N: int = 600,
    plant_params: Dict = None,
    twin_params: Dict = None,
    gain_ramp_steps: int = 30,
    gain_min_scale: float = 0.15,
    verbose: bool = False,
    store_trajectories: bool = False,
) -> ExperimentMetrics:
    """
    Run a single DDFS experiment with specified parameters.

    Args:
        T: Segment length
        L: Data window length (default: 0.6*T)
        alpha: Contraction rate
        mu: Growth bound
        gamma: Plant-twin mismatch bound (default from config)
        L_J: Jacobian Lipschitz constant (default from config)
        L_r: Remainder Lipschitz constant (default from config)
        C: Jacobian variation rate (default: L_J * v)
        x0_plant: Plant initial state (default from config)
        N: Total simulation steps
        plant_params: Custom plant parameters (default from config)
        twin_params: Custom twin parameters (default from config)
        gain_ramp_steps: Gain scheduling ramp steps
        gain_min_scale: Gain scheduling initial scale
        verbose: Print progress
        store_trajectories: Store full trajectory data in metrics

    Returns:
        ExperimentMetrics with results
    """
    metrics = ExperimentMetrics()

    # Use defaults from config if not specified
    if gamma is None:
        gamma = LIPSCHITZ['gamma']
    if L_J is None:
        L_J = LIPSCHITZ['L_J']
    if L_r is None:
        L_r = LIPSCHITZ['L_r']
    if C is None:
        C = LIPSCHITZ['C']
    if L is None:
        L = int(0.6 * T)
    if x0_plant is None:
        x0_plant = np.array(INITIAL_STATES['x_plant_0'])

    try:
        # Create systems
        if plant_params is not None:
            from dynamics.discretization import DiscreteSystem
            plant_arm = RobotArm2DoF(**plant_params)
            plant = DiscreteSystem(plant_arm.dynamics, SIM_PARAMS['dt'])
        else:
            plant = create_plant()

        if twin_params is not None:
            from dynamics.discretization import DiscreteSystem
            twin_arm = RobotArm2DoF(**twin_params)
            twin = DiscreteSystem(twin_arm.dynamics, SIM_PARAMS['dt'])
        else:
            twin = create_twin()

        # Generate nominal trajectory (always use standard twin)
        std_twin = create_twin()
        arm = RobotArm2DoF(**TWIN_PARAMS)
        x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
        x_goal = np.array(INITIAL_STATES['x_goal'])
        u_goal = np.array(INITIAL_STATES['u_goal'])

        x_nom, u_nom = generate_nominal_trajectory(std_twin, x_nom_0, x_goal, u_goal, N, arm=arm)

        # Create controller with custom parameters
        n, m = 4, 2
        controller = SegmentedFunnelController(
            twin=twin,
            x_nom=x_nom,
            u_nom=u_nom,
            T=T,
            L=L,
            alpha=alpha,
            mu=mu,
            eps_bar=SYNTH_PARAMS['eps_bar'],
            C=C,
            gamma=gamma,
            L_r=L_r,
            gain_ramp_steps=gain_ramp_steps,
            gain_min_scale=gain_min_scale,
        )

        # Initialize controller
        K0, P0 = design_initial_controller(twin, x_nom, u_nom, n, m)
        controller.initialize(K0, P0)

        # Run simulation
        results = run_simulation(
            plant, twin, x_nom, u_nom, x0_plant, controller, verbose=verbose
        )

        # Collect metrics
        metrics.terminal_deviation = np.linalg.norm(results.eta_trajectory[-1])
        metrics.max_deviation = np.max(np.linalg.norm(results.eta_trajectory, axis=1))
        metrics.rms_error = np.sqrt(np.mean(np.sum(results.eta_trajectory**2, axis=1)))

        violations, max_viol = count_constraint_violations(results.x_trajectory)
        metrics.constraint_violations = violations
        metrics.violation_magnitude = max_viol

        # SDP feasibility
        sdp_successes = sum(1 for r in results.sdp_results if r.get('status', '').startswith('optimal'))
        metrics.sdp_feasibility_rate = sdp_successes / max(len(results.sdp_results), 1)
        metrics.num_segments = len(results.sdp_results) + 1  # +1 for initial segment

        metrics.in_funnel_fraction = results.in_funnel_fraction
        metrics.success = True

        if store_trajectories:
            metrics.eta_trajectory = results.eta_trajectory
            metrics.x_trajectory = results.x_trajectory
            metrics.u_trajectory = results.u_trajectory

    except Exception as e:
        metrics.success = False
        metrics.error_message = str(e)
        if verbose:
            print(f"Experiment failed: {e}")

    return metrics


def run_baseline_experiment(
    x0_plant: np.ndarray = None,
    N: int = 600,
    plant_params: Dict = None,
    twin_params: Dict = None,
    verbose: bool = False,
) -> ExperimentMetrics:
    """
    Run baseline (twin controller on plant) experiment.

    Returns:
        ExperimentMetrics with baseline results
    """
    metrics = ExperimentMetrics()

    if x0_plant is None:
        x0_plant = np.array(INITIAL_STATES['x_plant_0'])

    try:
        # Create systems
        if plant_params is not None:
            from dynamics.discretization import DiscreteSystem
            plant_arm = RobotArm2DoF(**plant_params)
            plant = DiscreteSystem(plant_arm.dynamics, SIM_PARAMS['dt'])
        else:
            plant = create_plant()

        if twin_params is not None:
            from dynamics.discretization import DiscreteSystem
            twin_arm = RobotArm2DoF(**twin_params)
            twin = DiscreteSystem(twin_arm.dynamics, SIM_PARAMS['dt'])
        else:
            twin = create_twin()

        # Generate nominal trajectory (always use standard twin)
        std_twin = create_twin()
        arm = RobotArm2DoF(**TWIN_PARAMS)
        x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
        x_goal = np.array(INITIAL_STATES['x_goal'])
        u_goal = np.array(INITIAL_STATES['u_goal'])

        x_nom, u_nom = generate_nominal_trajectory(std_twin, x_nom_0, x_goal, u_goal, N, arm=arm)

        # Design baseline controller
        K_baseline = design_twin_controller(twin, x_goal, u_goal)

        # Run baseline
        results = run_baseline(plant, x_nom, u_nom, x0_plant, K_baseline, verbose=verbose)

        # Collect metrics
        metrics.terminal_deviation = np.linalg.norm(results.eta_trajectory[-1])
        metrics.max_deviation = np.max(np.linalg.norm(results.eta_trajectory, axis=1))
        metrics.rms_error = np.sqrt(np.mean(np.sum(results.eta_trajectory**2, axis=1)))

        violations, max_viol = count_constraint_violations(results.x_trajectory)
        metrics.constraint_violations = violations
        metrics.violation_magnitude = max_viol

        metrics.success = True

    except Exception as e:
        metrics.success = False
        metrics.error_message = str(e)

    return metrics
