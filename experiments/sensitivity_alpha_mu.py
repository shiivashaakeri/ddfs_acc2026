"""
§2. Sensitivity Analysis: Decay Rate α and Growth μ

Analyzes how Lyapunov decay rate α and cross-segment growth bound μ
interact. Provides selection guidelines based on feasibility conditions.

Addresses: AE, R6 concerns about parameter selection guidelines.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.base_runner import run_single_experiment, ExperimentMetrics


def compute_effective_decay(alpha: float, mu: float, T: int) -> float:
    """
    Compute effective decay rate: α̂ = α · μ^(1/T)

    For stability: α̂ < 1
    """
    return alpha * (mu ** (1 / T))


def compute_minimum_T(alpha: float, mu: float) -> float:
    """
    Compute minimum T for dwell time condition: T > -ln(μ) / ln(α)
    """
    if alpha >= 1.0 or mu <= 1.0:
        return 0
    return -np.log(mu) / np.log(alpha)


def run_alpha_mu_sensitivity(
    alpha_values: List[float] = None,
    mu_values: List[float] = None,
    T: int = 100,
    N: int = 600,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run α-μ sensitivity analysis (grid search).

    Args:
        alpha_values: List of α values to test
        mu_values: List of μ values to test
        T: Fixed segment length
        N: Total simulation steps
        verbose: Print progress

    Returns:
        DataFrame with results for each (α, μ) pair
    """
    if alpha_values is None:
        alpha_values = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    if mu_values is None:
        mu_values = [1.00, 1.01, 1.02, 1.05, 1.10]

    results = []
    total = len(alpha_values) * len(mu_values)
    count = 0

    if verbose:
        print(f"Testing {len(alpha_values)} α values × {len(mu_values)} μ values = {total} combinations")
        print(f"Fixed T={T}, N={N}")
        print("-" * 70)

    for alpha in alpha_values:
        for mu in mu_values:
            count += 1

            # Compute theoretical quantities
            eff_decay = compute_effective_decay(alpha, mu, T)
            min_T = compute_minimum_T(alpha, mu)
            theoretically_feasible = eff_decay < 1.0 and T > min_T

            if verbose:
                print(f"[{count}/{total}] α={alpha:.2f}, μ={mu:.2f}, "
                      f"α̂={eff_decay:.4f}, min_T={min_T:.1f}...", end=" ", flush=True)

            # Run experiment
            start_time = time.time()
            metrics = run_single_experiment(
                T=T,
                alpha=alpha,
                mu=mu,
                N=N,
                verbose=False,
            )
            elapsed = time.time() - start_time

            result = {
                'alpha': alpha,
                'mu': mu,
                'effective_decay': eff_decay,
                'min_T_required': min_T,
                'theoretically_feasible': theoretically_feasible,
                'terminal_deviation': metrics.terminal_deviation if metrics.success else np.nan,
                'max_deviation': metrics.max_deviation if metrics.success else np.nan,
                'rms_error': metrics.rms_error if metrics.success else np.nan,
                'constraint_violations': metrics.constraint_violations,
                'sdp_feasibility_rate': metrics.sdp_feasibility_rate,
                'in_funnel_fraction': metrics.in_funnel_fraction if metrics.success else 0,
                'success': metrics.success,
                'time': elapsed,
            }
            results.append(result)

            if verbose:
                if metrics.success:
                    print(f"terminal={metrics.terminal_deviation:.4f}, feasibility={metrics.sdp_feasibility_rate:.0%}")
                else:
                    print(f"FAILED: {metrics.error_message[:50]}")

    df = pd.DataFrame(results)

    if verbose:
        print("-" * 70)
        successful = df[df['success']]
        print(f"\nSuccess rate: {len(successful)}/{len(df)} ({100*len(successful)/len(df):.1f}%)")
        if len(successful) > 0:
            best = successful.loc[successful['terminal_deviation'].idxmin()]
            print(f"Best (α, μ): ({best['alpha']:.2f}, {best['mu']:.2f}) "
                  f"with terminal={best['terminal_deviation']:.4f}")

    return df


def plot_alpha_mu_heatmap(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create α-μ feasibility heatmap.

    Args:
        df: Results DataFrame from run_alpha_mu_sensitivity
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique values
    alphas = sorted(df['alpha'].unique())
    mus = sorted(df['mu'].unique())

    # Create matrices for heatmap
    feasibility_matrix = np.zeros((len(mus), len(alphas)))
    terminal_matrix = np.full((len(mus), len(alphas)), np.nan)

    for i, mu in enumerate(mus):
        for j, alpha in enumerate(alphas):
            row = df[(df['alpha'] == alpha) & (df['mu'] == mu)]
            if len(row) > 0:
                row = row.iloc[0]
                feasibility_matrix[i, j] = 1 if row['success'] and row['sdp_feasibility_rate'] > 0.8 else 0
                if row['success']:
                    terminal_matrix[i, j] = row['terminal_deviation']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Feasibility heatmap
    ax = axes[0]
    im = ax.imshow(feasibility_matrix, cmap='RdYlGn', aspect='auto',
                   extent=[alphas[0]-0.01, alphas[-1]+0.01, mus[-1]+0.02, mus[0]-0.02])
    ax.set_xlabel('Decay Rate α', fontsize=12)
    ax.set_ylabel('Growth Bound μ', fontsize=12)
    ax.set_title('Feasibility Map (Green=Feasible, Red=Infeasible)', fontsize=14)
    ax.set_xticks(alphas)
    ax.set_yticks(mus)

    # Add text annotations
    for i, mu in enumerate(mus):
        for j, alpha in enumerate(alphas):
            color = 'white' if feasibility_matrix[i, j] < 0.5 else 'black'
            ax.text(alpha, mu, f'{feasibility_matrix[i, j]:.0f}',
                    ha='center', va='center', color=color, fontsize=10)

    # Mark paper's choice
    ax.plot(0.98, 1.02, 'b*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    ax.annotate('Paper', (0.98, 1.02), xytext=(0.985, 1.04),
                fontsize=10, color='blue', weight='bold')

    # Right: Terminal deviation contours
    ax = axes[1]
    # Mask NaN values
    masked_terminal = np.ma.masked_invalid(terminal_matrix)

    if not np.all(np.isnan(terminal_matrix)):
        levels = np.linspace(np.nanmin(terminal_matrix), np.nanmax(terminal_matrix), 10)
        contour = ax.contourf(alphas, mus, masked_terminal, levels=levels, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Terminal Deviation')

    ax.set_xlabel('Decay Rate α', fontsize=12)
    ax.set_ylabel('Growth Bound μ', fontsize=12)
    ax.set_title('Terminal Deviation ||η(N)|| Contours', fontsize=14)

    # Mark paper's choice
    ax.plot(0.98, 1.02, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_mu_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/alpha_mu_heatmap.png")


def plot_alpha_mu_pareto(df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Create α-μ Pareto frontier plot.

    Args:
        df: Results DataFrame
        output_dir: Directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter successful runs
    successful = df[df['success'] & (df['sdp_feasibility_rate'] > 0.5)].copy()

    if len(successful) == 0:
        print("No successful runs for Pareto plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot: terminal deviation vs effective decay (inverse of funnel size proxy)
    scatter = ax.scatter(
        successful['terminal_deviation'],
        1 / successful['effective_decay'],  # Higher is better (more margin)
        c=successful['alpha'],
        cmap='coolwarm',
        s=100,
        alpha=0.7,
        edgecolors='black'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Decay Rate α', fontsize=12)

    # Annotate points with μ values
    for _, row in successful.iterrows():
        ax.annotate(f'μ={row["mu"]:.2f}',
                    (row['terminal_deviation'], 1/row['effective_decay']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Mark paper's choice
    paper_row = successful[(successful['alpha'] == 0.98) & (successful['mu'] == 1.02)]
    if len(paper_row) > 0:
        paper_row = paper_row.iloc[0]
        ax.plot(paper_row['terminal_deviation'], 1/paper_row['effective_decay'],
                'r*', markersize=20, markeredgecolor='black', markeredgewidth=2, label='Paper (α=0.98, μ=1.02)')

    ax.set_xlabel('Terminal Deviation ||η(N)||', fontsize=12)
    ax.set_ylabel('Stability Margin (1/α̂)', fontsize=12)
    ax.set_title('Trade-off: Tracking Error vs Stability Margin', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_mu_pareto.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir}/alpha_mu_pareto.png")


def main():
    """Run α-μ sensitivity analysis."""
    print("=" * 70)
    print("§2. Sensitivity Analysis: Decay Rate α and Growth μ")
    print("=" * 70)
    print()

    # Run experiments
    df = run_alpha_mu_sensitivity(
        alpha_values=[0.90, 0.92, 0.94, 0.96, 0.98, 0.99],
        mu_values=[1.00, 1.01, 1.02, 1.05, 1.10],
        T=100,
        verbose=True,
    )

    # Generate outputs
    print("\nGenerating plots...")
    plot_alpha_mu_heatmap(df)
    plot_alpha_mu_pareto(df)

    # Save raw data
    df.to_csv('outputs/alpha_mu_results.csv', index=False)
    print("Saved: outputs/alpha_mu_results.csv")

    # Print guidelines
    print("\n" + "=" * 70)
    print("Parameter Selection Guidelines:")
    print("=" * 70)
    print("1. Effective decay α̂ = α·μ^(1/T) must satisfy α̂ < 1")
    print("2. Smaller α (more aggressive decay) improves tracking but reduces feasibility margin")
    print("3. Larger μ allows funnel growth but requires smaller α to compensate")
    print("4. Recommended: α ∈ [0.94, 0.98], μ ∈ [1.01, 1.05] for typical systems")
    print("=" * 70)

    return df


if __name__ == "__main__":
    main()
