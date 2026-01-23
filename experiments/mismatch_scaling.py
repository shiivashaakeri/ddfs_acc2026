"""
§4. Plant-Twin Mismatch Scaling

Tests method robustness as plant-twin mismatch increases beyond baseline.
Finds the breaking point where the method fails.

Addresses: R5, AE concerns about mismatch robustness.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.params import PLANT_PARAMS, TWIN_PARAMS
from experiments.base_runner import run_single_experiment, run_baseline_experiment


def scale_mismatch(scale: float) -> Dict:
    """
    Create plant parameters with scaled mismatch from twin.

    Plant param = Twin param + scale × (Original_Plant - Original_Twin)

    Args:
        scale: Mismatch scale factor (1.0 = baseline)

    Returns:
        Dictionary of plant parameters
    """
    scaled_params = {}
    for key in TWIN_PARAMS:
        twin_val = TWIN_PARAMS[key]
        plant_val = PLANT_PARAMS[key]
        diff = plant_val - twin_val
        scaled_params[key] = twin_val + scale * diff

    return scaled_params


def compute_gamma_estimate(scale: float, base_gamma: float = 0.034) -> float:
    """
    Estimate γ (mismatch bound) based on mismatch scale.

    Simple linear scaling: γ = base_gamma × scale

    Args:
        scale: Mismatch scale factor
        base_gamma: Baseline gamma from config

    Returns:
        Scaled gamma value
    """
    return base_gamma * scale


def run_mismatch_scaling(
    scales: List[float] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run mismatch scaling experiments.

    Args:
        scales: List of mismatch scale factors to test
        verbose: Print progress

    Returns:
        DataFrame with results
    """
    if scales is None:
        scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    results = []

    # Run baseline once at scale=1.0
    if verbose:
        print("Running baseline comparison...")
    baseline = run_baseline_experiment(verbose=False)

    if verbose:
        print(f"Baseline terminal: {baseline.terminal_deviation:.4f}")
        print(f"\nTesting mismatch scales: {scales}")
        print("-" * 70)

    for scale in scales:
        # Create scaled plant parameters
        plant_params = scale_mismatch(scale)
        gamma_est = compute_gamma_estimate(scale)

        if verbose:
            print(f"Scale={scale:.1f}: γ≈{gamma_est:.4f}...", end=" ", flush=True)

        # Run proposed method with scaled gamma (add safety margin)
        metrics = run_single_experiment(
            plant_params=plant_params,
            gamma=gamma_est * 1.3,  # 30% safety margin
            verbose=False,
            store_trajectories=False,
        )

        # Run baseline with same plant
        baseline_scaled = run_baseline_experiment(
            plant_params=plant_params,
            verbose=False,
        )

        result = {
            'scale': scale,
            'gamma_estimate': gamma_est,
            'terminal_deviation': metrics.terminal_deviation if metrics.success else np.nan,
            'max_deviation': metrics.max_deviation if metrics.success else np.nan,
            'constraint_violations': metrics.constraint_violations,
            'sdp_feasibility_rate': metrics.sdp_feasibility_rate,
            'success': metrics.success,
            'baseline_terminal': baseline_scaled.terminal_deviation if baseline_scaled.success else np.nan,
            'baseline_max': baseline_scaled.max_deviation if baseline_scaled.success else np.nan,
            'beats_baseline': (metrics.terminal_deviation < baseline_scaled.terminal_deviation
                              if metrics.success and baseline_scaled.success else False),
        }
        results.append(result)

        if verbose:
            if metrics.success:
                ratio = metrics.terminal_deviation / baseline_scaled.terminal_deviation if baseline_scaled.terminal_deviation > 0 else np.nan
                print(f"terminal={metrics.terminal_deviation:.4f}, "
                      f"baseline={baseline_scaled.terminal_deviation:.4f}, "
                      f"ratio={ratio:.2f}")
            else:
                print("FAILED")

    return pd.DataFrame(results)


def plot_mismatch_scaling(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create mismatch scaling plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Terminal deviation vs scale
    ax = axes[0]

    successful = df[df['success']]
    failed = df[~df['success']]

    if len(successful) > 0:
        ax.plot(successful['scale'], successful['terminal_deviation'],
                'b-o', linewidth=2, markersize=8, label='Proposed Method')
        ax.plot(successful['scale'], successful['baseline_terminal'],
                'r--s', linewidth=2, markersize=8, label='Baseline')

    if len(failed) > 0:
        for _, row in failed.iterrows():
            ax.axvline(x=row['scale'], color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=failed['scale'].iloc[0], color='gray', linestyle=':',
                   alpha=0.7, label='Infeasible')

    ax.set_xlabel('Mismatch Scale Factor', fontsize=12)
    ax.set_ylabel('Terminal Deviation ||η(N)||', fontsize=12)
    ax.set_title('Tracking Error vs Plant-Twin Mismatch', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Right: Performance ratio
    ax = axes[1]

    if len(successful) > 0:
        ratio = successful['terminal_deviation'] / successful['baseline_terminal']
        colors = ['green' if r < 1 else 'red' for r in ratio]
        ax.bar(successful['scale'], ratio, color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)

    ax.set_xlabel('Mismatch Scale Factor', fontsize=12)
    ax.set_ylabel('Proposed / Baseline Ratio', fontsize=12)
    ax.set_title('Performance Ratio (< 1 = Proposed Better)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mismatch_scaling.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/mismatch_scaling.png")


def main():
    """Run mismatch scaling experiments."""
    print("=" * 70)
    print("§4. Plant-Twin Mismatch Scaling")
    print("=" * 70)
    print()

    df = run_mismatch_scaling(
        scales=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        verbose=True,
    )

    print("\nGenerating plots...")
    plot_mismatch_scaling(df)

    df.to_csv('outputs/mismatch_scaling.csv', index=False)
    print("Saved: outputs/mismatch_scaling.csv")

    # Find breaking point
    failed = df[~df['success']]
    if len(failed) > 0:
        breaking_point = failed['scale'].min()
        print(f"\nBreaking point: scale={breaking_point}")
    else:
        print(f"\nNo failures up to scale={df['scale'].max()}")

    return df


if __name__ == "__main__":
    main()
