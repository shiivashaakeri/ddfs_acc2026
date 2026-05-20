"""Run all reviewer experiments end-to-end."""

from __future__ import annotations

import sys

from experiments import (
    mismatch_scaling,
    reproduce_fig4,
    sensitivity_T,
    sensitivity_alpha_mu,
    timing_analysis,
)


def main():
    print("=" * 60)
    print("DDFS ACC 2026 — full reviewer experiments")
    print("=" * 60)

    for label, mod in [
        ("Fig. 4 reproduction", reproduce_fig4),
        ("Sensitivity to T", sensitivity_T),
        ("Sensitivity to (α, μ)", sensitivity_alpha_mu),
        ("Mismatch scaling", mismatch_scaling),
        ("Timing analysis", timing_analysis),
    ]:
        print(f"\n--- {label} ---")
        try:
            mod.main()
        except Exception as e:
            print(f"  experiment FAILED: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
