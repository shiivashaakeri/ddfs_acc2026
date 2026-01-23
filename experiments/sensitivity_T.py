"""
§1. Sensitivity Analysis: Segment Length T

Analyzes how segment length T affects tracking performance,
constraint satisfaction, and SDP feasibility.

Addresses: AE, R6 concerns about parameter selection guidelines.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.base_runner import run_single_experiment, run_baseline_experiment, ExperimentMetrics


def run_T_sensitivity(
    T_values: List[int] = None,
    L_ratio: float = 0.6,
    N: int = 600,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run T sensitivity analysis.

    Args:
        T_values: List of segment lengths to test
        L_ratio: Data window ratio (L = L_ratio * T)
        N: Total simulation steps
        verbose: Print progress

    Returns:
        DataFrame with results for each T value
    """
    if T_values is None:
        T_values = [25, 50, 75, 100, 125, 150, 200]

    results = []

    # Run baseline once
    if verbose:
        print("Running baseline experiment...")
    baseline = run_baseline_experiment(verbose=False)
    baseline_terminal = baseline.terminal_deviation
    baseline_max = baseline.max_deviation

    if verbose:
        print(f"Baseline: terminal={baseline_terminal:.4f}, max={baseline_max:.4f}")
        print(f"\nTesting T values: {T_values}")
        print("-" * 70)

    for T in T_values:
        L = int(L_ratio * T)
        num_segments = N // T

        if verbose:
            print(f"T={T:3d}, L={L:3d}, segments={num_segments}...", end=" ", flush=True)

        start_time = time.time()
        metrics = run_single_experiment(
            T=T,
            L=L,
            N=N,
            verbose=False,
        )
        elapsed = time.time() - start_time

        result = {
            'T': T,
            'L': L,
            'num_segments': num_segments,
            'terminal_deviation': metrics.terminal_deviation,
            'max_deviation': metrics.max_deviation,
            'rms_error': metrics.rms_error,
            'constraint_violations': metrics.constraint_violations,
            'violation_magnitude': metrics.violation_magnitude,
            'sdp_feasibility_rate': metrics.sdp_feasibility_rate,
            'in_funnel_fraction': metrics.in_funnel_fraction,
            'total_time': elapsed,
            'time_per_segment': elapsed / max(num_segments, 1),
            'success': metrics.success,
            'vs_baseline_terminal': metrics.terminal_deviation / baseline_terminal if baseline_terminal > 0 else np.nan,
            'vs_baseline_max': metrics.max_deviation / baseline_max if baseline_max > 0 else np.nan,
        }
        results.append(result)

        if verbose:
            status = "OK" if metrics.success else "FAIL"
            print(f"terminal={metrics.terminal_deviation:.4f}, max={metrics.max_deviation:.4f}, "
                  f"violations={metrics.constraint_violations}, time={elapsed:.2f}s [{status}]")

    df = pd.DataFrame(results)

    if verbose:
        print("-" * 70)
        print("\nSummary:")
        print(f"  Best terminal error: T={df.loc[df['terminal_deviation'].idxmin(), 'T']} "
              f"({df['terminal_deviation'].min():.4f})")
        print(f"  Best max error: T={df.loc[df['max_deviation'].idxmin(), 'T']} "
              f"({df['max_deviation'].min():.4f})")
        print(f"  Fewest violations: T={df.loc[df['constraint_violations'].idxmin(), 'T']} "
              f"({df['constraint_violations'].min()})")

    return df


def plot_T_sensitivity(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create T sensitivity plots.

    Args:
        df: Results DataFrame from run_T_sensitivity
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Terminal deviation vs T
    ax = axes[0, 0]
    ax.plot(df['T'], df['terminal_deviation'], 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0.047, color='r', linestyle='--', label='Target (4.725e-2)')
    ax.set_xlabel('Segment Length T', fontsize=12)
    ax.set_ylabel('Terminal Deviation ||η(N)||', fontsize=12)
    ax.set_title('Terminal Tracking Error vs T', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Top-right: Max deviation vs T
    ax = axes[0, 1]
    ax.plot(df['T'], df['max_deviation'], 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Segment Length T', fontsize=12)
    ax.set_ylabel('Maximum Deviation max||η(k)||', fontsize=12)
    ax.set_title('Maximum Tracking Error vs T', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Time per segment vs T
    ax = axes[1, 0]
    ax.plot(df['T'], df['time_per_segment'], 'm-^', linewidth=2, markersize=8)
    ax.set_xlabel('Segment Length T', fontsize=12)
    ax.set_ylabel('SDP Solve Time per Segment (s)', fontsize=12)
    ax.set_title('Computational Cost vs T', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Constraint violations vs T
    ax = axes[1, 1]
    ax.bar(df['T'], df['constraint_violations'], color='coral', edgecolor='darkred')
    ax.set_xlabel('Segment Length T', fontsize=12)
    ax.set_ylabel('Constraint Violations (count)', fontsize=12)
    ax.set_title('Constraint Violations vs T', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_T.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/sensitivity_T.png")


def create_T_guidelines_table(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create T selection guidelines table.

    Args:
        df: Results DataFrame
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create summary table
    table = df[['T', 'terminal_deviation', 'max_deviation', 'constraint_violations',
                'time_per_segment', 'sdp_feasibility_rate', 'vs_baseline_terminal']].copy()

    table.columns = ['T', 'Terminal Error', 'Max Error', 'Violations',
                     'Solve Time (s)', 'SDP Feasibility', 'vs Baseline']

    # Add recommendation column
    def get_recommendation(row):
        if row['Violations'] > 0:
            return 'Not recommended (violations)'
        elif row['Terminal Error'] < 0.02 and row['Solve Time (s)'] < 1.0:
            return 'RECOMMENDED'
        elif row['Terminal Error'] < 0.05:
            return 'Acceptable'
        else:
            return 'Poor tracking'

    table['Recommendation'] = table.apply(get_recommendation, axis=1)

    # Save to CSV
    table.to_csv(os.path.join(output_dir, 'sensitivity_T_table.csv'), index=False)
    print(f"Saved: {output_dir}/sensitivity_T_table.csv")

    # Print table
    print("\nT Selection Guidelines:")
    print(table.to_string(index=False))

    return table


def main():
    """Run T sensitivity analysis."""
    print("=" * 70)
    print("§1. Sensitivity Analysis: Segment Length T")
    print("=" * 70)
    print()

    # Run experiments
    df = run_T_sensitivity(
        T_values=[25, 50, 75, 100, 125, 150, 200],
        verbose=True,
    )

    # Generate outputs
    print("\nGenerating plots and tables...")
    plot_T_sensitivity(df)
    create_T_guidelines_table(df)

    # Save raw data
    df.to_csv('outputs/sensitivity_T_raw.csv', index=False)
    print("Saved: outputs/sensitivity_T_raw.csv")

    print("\n" + "=" * 70)
    print("§1 Complete")
    print("=" * 70)

    return df


if __name__ == "__main__":
    main()
