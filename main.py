#!/usr/bin/env python3
"""
DDFS ACC 2026 - Main Entry Point.

Robust Data-Driven Control for Nonlinear Systems
Using their Digital Twins and Quadratic Funnels

Authors: Shiva Shakeri and Mehran Mesbahi
University of Washington

Runs the complete data-driven funnel synthesis pipeline:
1. Create plant and twin systems
2. Generate nominal trajectory
3. Run proposed method (data-driven funnel control)
4. Run baseline (twin controller on plant)
5. Visualize and compare results

Expected output (from paper):
- 6 segments complete (N=600, T=100)
- All SDPs solve successfully
- Terminal deviation ||eta(N)|| ≈ 4.725 × 10^{-2}
- Proposed method outperforms baseline

Usage:
    python main.py
"""

from __future__ import annotations

import os
import numpy as np

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.params import (
    INITIAL_STATES,
    LIPSCHITZ,
    PLANT_PARAMS,
    SIM_PARAMS,
    SYNTH_PARAMS,
    TWIN_PARAMS,
)
from dynamics.digital_twin import create_plant, create_twin
from dynamics.robot_arm import RobotArm2DoF
from trajectory.nominal_planner import generate_nominal_trajectory
from simulation.closed_loop import run_simulation
from simulation.baseline import run_baseline, design_twin_controller, compare_methods
from visualization.trajectory_plots import (
    plot_tracking_results,
    plot_deviation_comparison,
)


def main():
    """Run the complete DDFS pipeline."""
    print("=" * 60)
    print("Data-Driven Funnel Synthesis for Nonlinear Systems")
    print("ACC 2026 Implementation")
    print("=" * 60)

    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    # 1. Create systems
    print("\n[1/6] Creating plant and twin systems...")
    plant = create_plant()
    twin = create_twin()

    # Create arm model for trajectory generation
    arm = RobotArm2DoF(**TWIN_PARAMS)

    # 2. Generate nominal trajectory
    print("\n[2/6] Generating nominal trajectory...")
    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    N = SIM_PARAMS["N"]

    x_nom, u_nom = generate_nominal_trajectory(
        twin, x_nom_0, x_goal, u_goal, N, arm=arm
    )

    print(f"  Trajectory: {N} steps ({N * SIM_PARAMS['dt']:.1f}s)")
    print(f"  Start: {x_nom[0]}")
    print(f"  Goal:  {x_nom[-1]}")

    # 3. Design baseline controller
    print("\n[3/6] Designing baseline controller...")
    K_baseline = design_twin_controller(twin, x_goal, u_goal)
    print(f"  K shape: {K_baseline.shape}")

    # 4. Run proposed method
    print("\n[4/6] Running proposed method (data-driven funnel control)...")
    x0_plant = np.array(INITIAL_STATES["x_plant_0"])
    print(f"  Plant initial state: {x0_plant}")
    print(f"  Initial deviation ||eta(0)||: {np.linalg.norm(x0_plant - x_nom[0]):.4f}")

    results = run_simulation(
        plant, twin, x_nom, u_nom, x0_plant, verbose=True
    )

    # 5. Run baseline
    print("\n[5/6] Running baseline (twin controller on plant)...")
    baseline = run_baseline(plant, x_nom, u_nom, x0_plant, K_baseline, verbose=True)

    # 6. Compare and visualize
    print("\n[6/6] Comparing results and generating figures...")
    comparison = compare_methods(results, baseline)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Segments completed: {len(results.P_history)}")
    print(f"  All SDPs succeeded: {results.all_sdps_succeeded}")
    print(f"")
    print(f"  PROPOSED METHOD:")
    print(f"    Terminal ||eta(N)||: {results.eta_final_norm:.4e}")
    print(f"    Maximum ||eta||:     {results.max_eta_norm:.4f}")
    print(f"    In-funnel fraction:  {results.in_funnel_fraction:.2%}")
    print(f"")
    print(f"  BASELINE:")
    print(f"    Terminal ||eta(N)||: {baseline.eta_final_norm:.4e}")
    print(f"    Maximum ||eta||:     {baseline.max_eta_norm:.4f}")
    print(f"")
    print(f"  IMPROVEMENT:")
    print(f"    Final error ratio:   {comparison['improvement_final']:.2f}x")
    print(f"    Max error ratio:     {comparison['improvement_max']:.2f}x")
    print("=" * 60)

    # Generate figures
    print("\nGenerating figures...")

    # Figure 4: 4-panel tracking comparison
    fig1 = plot_tracking_results(results, baseline, dt=SIM_PARAMS["dt"])
    fig1.savefig("figures/tracking_results.png", dpi=150, bbox_inches="tight")
    print("  Saved: figures/tracking_results.png")

    # Deviation comparison
    fig2 = plot_deviation_comparison(results, baseline, dt=SIM_PARAMS["dt"])
    fig2.savefig("figures/deviation_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved: figures/deviation_comparison.png")

    plt.close("all")

    # Report SDP results
    print("\nSDP Results per segment:")
    for sdp_result in results.sdp_results:
        print(f"  Segment {sdp_result['segment']}: {sdp_result['status']}, "
              f"beta={sdp_result['beta']:.2e}, rank={sdp_result['rank']}")

    # Final verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    target_terminal = 4.725e-2
    print(f"  Target terminal ||eta||:  {target_terminal:.4e}")
    print(f"  Achieved terminal ||eta||: {results.eta_final_norm:.4e}")

    if results.eta_final_norm < 0.1:
        print("  Status: PASS (reasonable terminal deviation)")
    else:
        print("  Status: NEEDS INVESTIGATION (large terminal deviation)")

    if comparison["improvement_final"] > 1.0:
        print(f"  Proposed outperforms baseline: YES ({comparison['improvement_final']:.2f}x)")
    else:
        print(f"  Proposed outperforms baseline: NO ({comparison['improvement_final']:.2f}x)")

    print("=" * 60)

    return results, baseline, comparison


if __name__ == "__main__":
    results, baseline, comparison = main()
