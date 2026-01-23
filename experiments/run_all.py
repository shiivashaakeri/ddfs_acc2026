#!/usr/bin/env python3
"""
Run all DDFS reviewer experiments.

Usage:
    python experiments/run_all.py              # Run all experiments
    python experiments/run_all.py --quick      # Run quick subset for testing
    python experiments/run_all.py --section 1  # Run specific section
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_section_1(quick: bool = False):
    """§1. T Sensitivity Analysis"""
    from experiments.sensitivity_T import run_T_sensitivity, plot_T_sensitivity, create_T_guidelines_table

    if quick:
        T_values = [50, 100, 150]
    else:
        T_values = [25, 50, 75, 100, 125, 150, 200]

    df = run_T_sensitivity(T_values=T_values, verbose=True)
    plot_T_sensitivity(df)
    create_T_guidelines_table(df)
    df.to_csv('outputs/sensitivity_T_raw.csv', index=False)

    return df


def run_section_2(quick: bool = False):
    """§2. α-μ Sensitivity Analysis"""
    from experiments.sensitivity_alpha_mu import (
        run_alpha_mu_sensitivity, plot_alpha_mu_heatmap, plot_alpha_mu_pareto
    )

    if quick:
        alpha_values = [0.94, 0.96, 0.98]
        mu_values = [1.01, 1.02, 1.05]
    else:
        alpha_values = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
        mu_values = [1.00, 1.01, 1.02, 1.05, 1.10]

    df = run_alpha_mu_sensitivity(
        alpha_values=alpha_values,
        mu_values=mu_values,
        verbose=True,
    )
    plot_alpha_mu_heatmap(df)
    plot_alpha_mu_pareto(df)
    df.to_csv('outputs/alpha_mu_results.csv', index=False)

    return df


def run_section_3(quick: bool = False):
    """§3. Lipschitz Sensitivity Analysis"""
    from experiments.sensitivity_lipschitz import (
        run_all_lipschitz_experiments, plot_lipschitz_sensitivity,
        plot_conservatism_analysis, create_robustness_summary
    )

    results = run_all_lipschitz_experiments(verbose=True)
    plot_lipschitz_sensitivity(results)
    plot_conservatism_analysis(results)
    create_robustness_summary(results)

    for param, df in results.items():
        df.to_csv(f'outputs/lipschitz_{param}_raw.csv', index=False)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run DDFS reviewer experiments')
    parser.add_argument('--quick', action='store_true', help='Run quick subset for testing')
    parser.add_argument('--section', type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='Run specific section only')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)

    print("=" * 70)
    print("DDFS ACC 2026 Reviewer Experiments")
    print("=" * 70)
    if args.quick:
        print("Running in QUICK mode (reduced parameter ranges)")
    print()

    total_start = time.time()

    sections = {
        1: ("§1. T Sensitivity", run_section_1),
        2: ("§2. α-μ Sensitivity", run_section_2),
        3: ("§3. Lipschitz Sensitivity", run_section_3),
    }

    if args.section:
        sections_to_run = {args.section: sections[args.section]}
    else:
        sections_to_run = sections

    results = {}
    for num, (name, func) in sections_to_run.items():
        print("\n" + "=" * 70)
        print(f"{name}")
        print("=" * 70)

        section_start = time.time()
        try:
            results[num] = func(quick=args.quick)
            elapsed = time.time() - section_start
            print(f"\n{name} completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n{name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"All experiments completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("Outputs saved to: outputs/")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
