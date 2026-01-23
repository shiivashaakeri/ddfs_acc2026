"""
§5. Unmodeled Dynamics & External Disturbances

Tests robustness to unmodeled dynamics (R2) and external disturbances (R7)
not captured by plant-twin mismatch.

Addresses: R2, R7 concerns about unmodeled effects.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.params import INITIAL_STATES, SIM_PARAMS, TWIN_PARAMS
from dynamics.digital_twin import create_plant, create_twin
from dynamics.robot_arm import RobotArm2DoF
from trajectory.nominal_planner import generate_nominal_trajectory
from control.segmented_controller import create_controller_from_params
from simulation.baseline import design_twin_controller


def run_with_disturbance(
    disturbance_func: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    disturbance_name: str,
    verbose: bool = False,
) -> Dict:
    """
    Run simulation with external disturbance added to control input.

    Args:
        disturbance_func: Function (k, x, u) -> d that returns disturbance
        disturbance_name: Name of disturbance for logging
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    from dynamics.discretization import DiscreteSystem

    # Setup
    plant = create_plant()
    twin = create_twin()
    arm = RobotArm2DoF(**TWIN_PARAMS)

    x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
    x_goal = np.array(INITIAL_STATES['x_goal'])
    u_goal = np.array(INITIAL_STATES['u_goal'])
    N = SIM_PARAMS['N']

    x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)

    # Create controller
    controller = create_controller_from_params(twin, x_nom, u_nom)
    x0_plant = np.array(INITIAL_STATES['x_plant_0'])

    # Run simulation with disturbance
    n, m = 4, 2
    x_traj = np.zeros((N + 1, n))
    u_traj = np.zeros((N, m))
    eta_traj = np.zeros((N + 1, n))

    x = x0_plant.copy()
    x_traj[0] = x
    eta_traj[0] = x - x_nom[0]

    U_MIN, U_MAX = -40.0, 40.0

    try:
        for k in range(N):
            # Get control
            u = controller.compute_control(k, x)

            # Add disturbance
            d = disturbance_func(k, x, u)
            u_disturbed = u + d

            # Saturate
            u_disturbed = np.clip(u_disturbed, U_MIN, U_MAX)
            u_traj[k] = u_disturbed

            # Step plant
            x_next = plant.step(x, u_disturbed)

            # Check for numerical issues
            if not np.all(np.isfinite(x_next)):
                raise ValueError(f"Numerical overflow at k={k}")

            x = x_next
            x_traj[k + 1] = x
            eta_traj[k + 1] = x - x_nom[k + 1]

            # End of segment update
            if (k + 1) % SIM_PARAMS['T'] == 0 and k < N - 1:
                controller.end_of_segment_update(k)

        success = True
        error_msg = ""

    except Exception as e:
        success = False
        error_msg = str(e)

    # Compute metrics
    if success:
        terminal_dev = np.linalg.norm(eta_traj[-1])
        max_dev = np.max(np.linalg.norm(eta_traj, axis=1))
    else:
        terminal_dev = np.nan
        max_dev = np.nan

    return {
        'name': disturbance_name,
        'terminal_deviation': terminal_dev,
        'max_deviation': max_dev,
        'success': success,
        'error': error_msg,
    }


def run_baseline_with_disturbance(
    disturbance_func: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
) -> Dict:
    """Run baseline with disturbance."""
    plant = create_plant()
    twin = create_twin()
    arm = RobotArm2DoF(**TWIN_PARAMS)

    x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
    x_goal = np.array(INITIAL_STATES['x_goal'])
    u_goal = np.array(INITIAL_STATES['u_goal'])
    N = SIM_PARAMS['N']

    x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)
    K = design_twin_controller(twin, x_goal, u_goal)
    x0_plant = np.array(INITIAL_STATES['x_plant_0'])

    n, m = 4, 2
    x = x0_plant.copy()
    eta_traj = np.zeros((N + 1, n))
    eta_traj[0] = x - x_nom[0]

    U_MIN, U_MAX = -40.0, 40.0

    try:
        for k in range(N):
            eta = x - x_nom[k]
            u = u_nom[k] + K @ eta
            d = disturbance_func(k, x, u)
            u_disturbed = np.clip(u + d, U_MIN, U_MAX)
            x = plant.step(x, u_disturbed)
            eta_traj[k + 1] = x - x_nom[k + 1]

        return {
            'terminal_deviation': np.linalg.norm(eta_traj[-1]),
            'max_deviation': np.max(np.linalg.norm(eta_traj, axis=1)),
            'success': True,
        }
    except Exception:
        return {'terminal_deviation': np.nan, 'max_deviation': np.nan, 'success': False}


def run_disturbance_experiments(verbose: bool = True) -> pd.DataFrame:
    """
    Run all disturbance experiments.
    """
    results = []
    dt = SIM_PARAMS['dt']

    # Define disturbance scenarios
    disturbances = [
        # No disturbance (baseline comparison)
        ("None", lambda k, x, u: np.zeros(2)),

        # Constant torque disturbances
        ("Constant [1, 0] Nm", lambda k, x, u: np.array([1.0, 0.0])),
        ("Constant [2, 0] Nm", lambda k, x, u: np.array([2.0, 0.0])),
        ("Constant [1, 1] Nm", lambda k, x, u: np.array([1.0, 1.0])),
        ("Constant [2, 2] Nm", lambda k, x, u: np.array([2.0, 2.0])),

        # Sinusoidal disturbances (1 Hz)
        ("Sinusoid A=0.5 Nm", lambda k, x, u: 0.5 * np.sin(2 * np.pi * 1.0 * k * dt) * np.ones(2)),
        ("Sinusoid A=1.0 Nm", lambda k, x, u: 1.0 * np.sin(2 * np.pi * 1.0 * k * dt) * np.ones(2)),
        ("Sinusoid A=2.0 Nm", lambda k, x, u: 2.0 * np.sin(2 * np.pi * 1.0 * k * dt) * np.ones(2)),

        # Random disturbances
        ("Random σ=0.5 Nm", lambda k, x, u: np.random.randn(2) * 0.5),
        ("Random σ=1.0 Nm", lambda k, x, u: np.random.randn(2) * 1.0),
        ("Random σ=2.0 Nm", lambda k, x, u: np.random.randn(2) * 2.0),
    ]

    if verbose:
        print(f"Testing {len(disturbances)} disturbance scenarios...")
        print("-" * 70)

    for name, dist_func in disturbances:
        if verbose:
            print(f"{name:25s}...", end=" ", flush=True)

        # Set random seed for reproducibility
        np.random.seed(42)

        # Run proposed method
        proposed = run_with_disturbance(dist_func, name, verbose=False)

        # Run baseline
        np.random.seed(42)
        baseline = run_baseline_with_disturbance(dist_func)

        result = {
            'disturbance': name,
            'proposed_terminal': proposed['terminal_deviation'],
            'proposed_max': proposed['max_deviation'],
            'proposed_success': proposed['success'],
            'baseline_terminal': baseline['terminal_deviation'],
            'baseline_max': baseline['max_deviation'],
            'baseline_success': baseline['success'],
        }

        if proposed['success'] and baseline['success']:
            result['beats_baseline'] = proposed['terminal_deviation'] < baseline['terminal_deviation']
            result['ratio'] = proposed['terminal_deviation'] / baseline['terminal_deviation']
        else:
            result['beats_baseline'] = False
            result['ratio'] = np.nan

        results.append(result)

        if verbose:
            if proposed['success']:
                print(f"proposed={proposed['terminal_deviation']:.4f}, "
                      f"baseline={baseline['terminal_deviation']:.4f}, "
                      f"ratio={result['ratio']:.2f}")
            else:
                print(f"FAILED: {proposed['error'][:40]}")

    return pd.DataFrame(results)


def plot_disturbance_results(df: pd.DataFrame, output_dir: str = "outputs"):
    """Create disturbance rejection comparison plot."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter successful runs
    successful = df[df['proposed_success'] & df['baseline_success']]

    if len(successful) > 0:
        x = np.arange(len(successful))
        width = 0.35

        bars1 = ax.bar(x - width/2, successful['proposed_terminal'], width,
                       label='Proposed Method', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, successful['baseline_terminal'], width,
                       label='Baseline', color='coral', edgecolor='black')

        ax.set_xlabel('Disturbance Type', fontsize=12)
        ax.set_ylabel('Terminal Deviation ||η(N)||', fontsize=12)
        ax.set_title('Disturbance Rejection Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(successful['disturbance'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disturbance_rejection.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/disturbance_rejection.png")


def main():
    """Run disturbance experiments."""
    print("=" * 70)
    print("§5. Unmodeled Dynamics & External Disturbances")
    print("=" * 70)
    print()

    df = run_disturbance_experiments(verbose=True)

    print("\nGenerating plots...")
    plot_disturbance_results(df)

    df.to_csv('outputs/disturbance_results.csv', index=False)
    print("Saved: outputs/disturbance_results.csv")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    successful = df[df['proposed_success']]
    if len(successful) > 0:
        wins = successful['beats_baseline'].sum()
        print(f"  Proposed beats baseline in {wins}/{len(successful)} cases")
        print(f"  Average ratio: {successful['ratio'].mean():.2f}")

    return df


if __name__ == "__main__":
    main()
