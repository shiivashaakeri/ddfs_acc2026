"""
§6. Multiple Initial Conditions

Demonstrates method works across different initial deviations,
not just the single case in the paper.

Addresses: AE concern about generalization.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.params import INITIAL_STATES, STATE_BOUNDS
from experiments.base_runner import run_single_experiment, run_baseline_experiment


def sample_initial_conditions(
    magnitudes: List[float],
    n_directions: int = 20,
    seed: int = 42,
) -> List[Tuple[np.ndarray, float]]:
    """
    Sample initial conditions at various magnitudes.

    Args:
        magnitudes: List of ||η(0)|| magnitudes to test
        n_directions: Number of random directions per magnitude
        seed: Random seed

    Returns:
        List of (x0_plant, magnitude) tuples
    """
    np.random.seed(seed)

    x_nom_0 = np.array(INITIAL_STATES['x_nom_0'])
    n = len(x_nom_0)

    # State bounds for feasibility check
    bounds = np.array([
        [STATE_BOUNDS['q1'][0], STATE_BOUNDS['q1'][1]],
        [STATE_BOUNDS['q2'][0], STATE_BOUNDS['q2'][1]],
        [STATE_BOUNDS['dq1'][0], STATE_BOUNDS['dq1'][1]],
        [STATE_BOUNDS['dq2'][0], STATE_BOUNDS['dq2'][1]],
    ])

    samples = []

    for mag in magnitudes:
        for _ in range(n_directions):
            # Random direction
            direction = np.random.randn(n)
            direction = direction / np.linalg.norm(direction)

            # Initial state
            eta_0 = mag * direction
            x0 = x_nom_0 + eta_0

            # Check feasibility
            feasible = np.all(x0 >= bounds[:, 0]) and np.all(x0 <= bounds[:, 1])

            if feasible:
                samples.append((x0, mag))

    return samples


def run_initial_condition_experiments(
    magnitudes: List[float] = None,
    n_directions: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run experiments with multiple initial conditions.

    Args:
        magnitudes: List of initial deviation magnitudes
        n_directions: Number of directions per magnitude
        verbose: Print progress

    Returns:
        DataFrame with results
    """
    if magnitudes is None:
        magnitudes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    samples = sample_initial_conditions(magnitudes, n_directions)

    if verbose:
        print(f"Testing {len(samples)} initial conditions")
        print(f"Magnitudes: {magnitudes}")
        print("-" * 70)

    results = []

    for i, (x0, mag) in enumerate(samples):
        if verbose:
            print(f"[{i+1}/{len(samples)}] ||η(0)||={mag:.1f}...", end=" ", flush=True)

        # Run proposed method
        metrics = run_single_experiment(
            x0_plant=x0,
            verbose=False,
        )

        # Run baseline
        baseline = run_baseline_experiment(
            x0_plant=x0,
            verbose=False,
        )

        result = {
            'initial_magnitude': mag,
            'x0': x0.tolist(),
            'terminal_deviation': metrics.terminal_deviation if metrics.success else np.nan,
            'max_deviation': metrics.max_deviation if metrics.success else np.nan,
            'constraint_violations': metrics.constraint_violations,
            'success': metrics.success,
            'baseline_terminal': baseline.terminal_deviation if baseline.success else np.nan,
            'beats_baseline': (metrics.terminal_deviation < baseline.terminal_deviation
                              if metrics.success and baseline.success else False),
        }
        results.append(result)

        if verbose:
            if metrics.success:
                print(f"terminal={metrics.terminal_deviation:.4f}, "
                      f"violations={metrics.constraint_violations}")
            else:
                print("FAILED")

    return pd.DataFrame(results)


def plot_initial_conditions(df: pd.DataFrame, output_dir: str = "outputs"):
    """Create initial conditions analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot of initial vs terminal deviation
    ax = axes[0]

    successful = df[df['success']]
    failed = df[~df['success']]

    if len(successful) > 0:
        colors = ['green' if b else 'blue' for b in successful['beats_baseline']]
        ax.scatter(successful['initial_magnitude'], successful['terminal_deviation'],
                   c=colors, s=60, alpha=0.7, edgecolors='black')

    if len(failed) > 0:
        ax.scatter(failed['initial_magnitude'],
                   np.full(len(failed), df['terminal_deviation'].max() * 1.1),
                   c='red', marker='x', s=100, linewidths=2, label='Failed')

    ax.set_xlabel('Initial Deviation ||η(0)||', fontsize=12)
    ax.set_ylabel('Terminal Deviation ||η(N)||', fontsize=12)
    ax.set_title('Terminal vs Initial Deviation', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Beats baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Below baseline'),
        Line2D([0], [0], marker='x', color='red', markersize=10, linewidth=2, label='Failed'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    # Right: Success rate by magnitude
    ax = axes[1]

    magnitudes = sorted(df['initial_magnitude'].unique())
    success_rates = []
    for mag in magnitudes:
        subset = df[df['initial_magnitude'] == mag]
        rate = subset['success'].mean() * 100
        success_rates.append(rate)

    colors = ['green' if r == 100 else 'orange' if r > 50 else 'red' for r in success_rates]
    ax.bar(magnitudes, success_rates, color=colors, edgecolor='black', alpha=0.7, width=0.4)

    ax.set_xlabel('Initial Deviation ||η(0)||', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Initial Deviation Magnitude', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_conditions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/initial_conditions.png")


def main():
    """Run initial conditions experiments."""
    print("=" * 70)
    print("§6. Multiple Initial Conditions")
    print("=" * 70)
    print()

    df = run_initial_condition_experiments(
        magnitudes=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        n_directions=10,
        verbose=True,
    )

    print("\nGenerating plots...")
    plot_initial_conditions(df)

    df.to_csv('outputs/initial_conditions.csv', index=False)
    print("Saved: outputs/initial_conditions.csv")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary by magnitude:")
    for mag in sorted(df['initial_magnitude'].unique()):
        subset = df[df['initial_magnitude'] == mag]
        success_rate = subset['success'].mean() * 100
        if subset['success'].any():
            mean_terminal = subset.loc[subset['success'], 'terminal_deviation'].mean()
            print(f"  ||η(0)||={mag:.1f}: success={success_rate:.0f}%, mean_terminal={mean_terminal:.4f}")
        else:
            print(f"  ||η(0)||={mag:.1f}: success={success_rate:.0f}%")

    return df


if __name__ == "__main__":
    main()
