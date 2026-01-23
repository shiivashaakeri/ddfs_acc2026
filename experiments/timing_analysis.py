"""
§7. Computational Complexity Analysis

Provides computational complexity data and real-time feasibility assessment.

Addresses: R6 concern about computational requirements.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.params import INITIAL_STATES, SIM_PARAMS, TWIN_PARAMS, LIPSCHITZ, SYNTH_PARAMS
from dynamics.digital_twin import create_plant, create_twin
from dynamics.robot_arm import RobotArm2DoF
from trajectory.nominal_planner import generate_nominal_trajectory
from control.segmented_controller import SegmentedFunnelController, design_initial_controller
from data_driven import build_data_matrices, compute_beta, SegmentManager
from synthesis.ellipsoid_bounds import compute_mvie_state, compute_mvie_input, get_bounds_from_config
from synthesis.sdp_solver import solve_funnel_sdp


def time_component(func, *args, n_trials: int = 3, **kwargs) -> float:
    """Time a function over multiple trials."""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)


def run_timing_breakdown(verbose: bool = True) -> Dict:
    """
    Measure timing for each component of the algorithm.
    """
    # Setup
    twin = create_twin()
    arm = RobotArm2DoF(**TWIN_PARAMS)

    x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
    x_goal = np.array(INITIAL_STATES['x_goal'])
    u_goal = np.array(INITIAL_STATES['u_goal'])
    N = SIM_PARAMS['N']
    T = SIM_PARAMS['T']
    L = SIM_PARAMS['L']
    n, m = 4, 2

    x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)

    # Create synthetic data for testing
    segment_manager = SegmentManager(T, L, n, m)
    np.random.seed(42)
    eta_log = 0.1 * np.random.randn(T + 1, n)
    xi_log = 0.1 * np.random.randn(T, m)

    x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()

    timings = {}

    if verbose:
        print("Timing individual components...")
        print("-" * 50)

    # 1. Data matrix construction
    def build_data():
        return build_data_matrices(eta_log, xi_log, 0, segment_manager)

    t = time_component(build_data, n_trials=10)
    timings['data_matrix_construction'] = t
    if verbose:
        print(f"  Data matrix construction: {t*1000:.3f} ms")

    H, H_plus, Xi = build_data()

    # 2. Beta computation
    data_window = segment_manager.get_data_window_indices(0)
    k_i = segment_manager.get_segment_start(0)

    def compute_beta_timed():
        return compute_beta(eta_log, xi_log, data_window, k_i,
                           LIPSCHITZ['C'], LIPSCHITZ['gamma'], LIPSCHITZ['L_r'])

    t = time_component(compute_beta_timed, n_trials=10)
    timings['beta_computation'] = t
    if verbose:
        print(f"  Beta computation: {t*1000:.3f} ms")

    beta = 1.0  # Use regularized value

    # 3. Ellipsoid bounds (MVIE)
    def compute_bounds():
        P_min = compute_mvie_state(x_nom[T], x_lower, x_upper)
        R_max = compute_mvie_input(u_nom[T], u_lower, u_upper)
        return P_min, R_max

    t = time_component(compute_bounds, n_trials=10)
    timings['ellipsoid_bounds'] = t
    if verbose:
        print(f"  Ellipsoid bounds (MVIE): {t*1000:.3f} ms")

    P_min, R_max = compute_bounds()

    # 4. Initial controller design
    def design_controller():
        return design_initial_controller(twin, x_nom, u_nom, n, m)

    t = time_component(design_controller, n_trials=5)
    timings['initial_controller'] = t
    if verbose:
        print(f"  Initial controller (LQR): {t*1000:.3f} ms")

    K0, P0 = design_controller()

    # 5. SDP solve (main computational bottleneck)
    T_tilde = 2 * T - L - 1

    def solve_sdp():
        return solve_funnel_sdp(
            H, H_plus, Xi, beta,
            P_min, R_max,
            SYNTH_PARAMS['alpha'], LIPSCHITZ['C'], T_tilde,
            n, m,
            P_prev=P0,
            mu=SYNTH_PARAMS['mu'],
            verbose=False,
        )

    t = time_component(solve_sdp, n_trials=3)
    timings['sdp_solve'] = t
    if verbose:
        print(f"  SDP solve: {t*1000:.1f} ms")

    # 6. Control computation (per-step)
    controller = SegmentedFunnelController(twin, x_nom, u_nom, T, L)
    controller.initialize(K0, P0)
    x_test = x_nom[0] + 0.1 * np.random.randn(n)

    def compute_control():
        return controller.K @ (x_test - x_nom[0])

    t = time_component(compute_control, n_trials=1000)
    timings['control_computation'] = t
    if verbose:
        print(f"  Control computation (per step): {t*1e6:.3f} μs")

    # Total per-segment overhead
    timings['total_per_segment'] = (
        timings['data_matrix_construction'] +
        timings['beta_computation'] +
        timings['ellipsoid_bounds'] +
        timings['sdp_solve']
    )

    if verbose:
        print("-" * 50)
        print(f"  Total per-segment overhead: {timings['total_per_segment']*1000:.1f} ms")

    return timings


def run_scaling_analysis(
    L_values: List[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Analyze how SDP solve time scales with data window size L.
    """
    if L_values is None:
        L_values = [20, 40, 60, 80, 100]

    # Setup
    twin = create_twin()
    arm = RobotArm2DoF(**TWIN_PARAMS)

    x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
    x_goal = np.array(INITIAL_STATES['x_goal'])
    u_goal = np.array(INITIAL_STATES['u_goal'])
    N = SIM_PARAMS['N']
    T = SIM_PARAMS['T']
    n, m = 4, 2

    x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)

    x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
    P_min = compute_mvie_state(x_nom[T], x_lower, x_upper)
    R_max = compute_mvie_input(u_nom[T], u_lower, u_upper)
    K0, P0 = design_initial_controller(twin, x_nom, u_nom, n, m)

    results = []

    if verbose:
        print(f"\nTesting SDP scaling with L = {L_values}")
        print("-" * 50)

    for L in L_values:
        # Create data matrices of size L
        segment_manager = SegmentManager(T, L, n, m)
        np.random.seed(42)
        eta_log = 0.1 * np.random.randn(T + 1, n)
        xi_log = 0.1 * np.random.randn(T, m)

        H, H_plus, Xi = build_data_matrices(eta_log, xi_log, 0, segment_manager)
        T_tilde = 2 * T - L - 1

        if verbose:
            print(f"  L={L}: H shape={H.shape}...", end=" ", flush=True)

        # Time SDP
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = solve_funnel_sdp(
                H, H_plus, Xi, 1.0,
                P_min, R_max,
                SYNTH_PARAMS['alpha'], LIPSCHITZ['C'], T_tilde,
                n, m,
                P_prev=P0,
                verbose=False,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        results.append({
            'L': L,
            'data_matrix_size': L,
            'sdp_time_ms': avg_time * 1000,
            'feasible': result.success,
        })

        if verbose:
            print(f"SDP time={avg_time*1000:.1f} ms")

    return pd.DataFrame(results)


def plot_timing_results(timings: Dict, scaling_df: pd.DataFrame, output_dir: str = "outputs"):
    """Create timing analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Timing breakdown pie chart
    ax = axes[0]

    components = ['data_matrix_construction', 'beta_computation', 'ellipsoid_bounds', 'sdp_solve']
    labels = ['Data Matrix', 'β Computation', 'Ellipsoid Bounds', 'SDP Solve']
    sizes = [timings[c] for c in components]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=(0, 0, 0, 0.1))
    ax.set_title('Computation Time Breakdown (per segment)', fontsize=14)

    # Right: SDP time vs L
    ax = axes[1]

    ax.plot(scaling_df['L'], scaling_df['sdp_time_ms'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Data Window Size L', fontsize=12)
    ax.set_ylabel('SDP Solve Time (ms)', fontsize=12)
    ax.set_title('SDP Computation Time vs Data Window Size', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add real-time threshold line
    segment_duration_ms = SIM_PARAMS['T'] * SIM_PARAMS['dt'] * 1000
    ax.axhline(y=segment_duration_ms, color='red', linestyle='--', linewidth=2,
               label=f'Segment duration ({segment_duration_ms:.0f} ms)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/timing_analysis.png")


def main():
    """Run timing analysis."""
    print("=" * 70)
    print("§7. Computational Complexity Analysis")
    print("=" * 70)
    print()

    # Timing breakdown
    timings = run_timing_breakdown(verbose=True)

    # Scaling analysis
    scaling_df = run_scaling_analysis(
        L_values=[20, 40, 60, 80, 100],
        verbose=True,
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_timing_results(timings, scaling_df)

    # Save results
    timing_df = pd.DataFrame([timings])
    timing_df.to_csv('outputs/timing_breakdown.csv', index=False)
    scaling_df.to_csv('outputs/timing_scaling.csv', index=False)
    print("Saved: outputs/timing_breakdown.csv, outputs/timing_scaling.csv")

    # Real-time feasibility assessment
    print("\n" + "=" * 70)
    print("Real-Time Feasibility Assessment:")
    print("=" * 70)

    segment_duration = SIM_PARAMS['T'] * SIM_PARAMS['dt']
    sdp_time = timings['sdp_solve']

    print(f"  Segment duration: {segment_duration:.2f} s ({segment_duration*1000:.0f} ms)")
    print(f"  SDP solve time: {sdp_time:.3f} s ({sdp_time*1000:.1f} ms)")
    print(f"  Ratio: {sdp_time/segment_duration:.1%}")

    if sdp_time < segment_duration:
        print(f"  Status: REAL-TIME FEASIBLE ✓")
        print(f"  Margin: {(segment_duration - sdp_time)*1000:.0f} ms remaining")
    else:
        print(f"  Status: NOT REAL-TIME FEASIBLE ✗")
        print(f"  Required speedup: {sdp_time/segment_duration:.1f}x")

    return timings, scaling_df


if __name__ == "__main__":
    main()
