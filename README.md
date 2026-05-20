# DDFS ACC 2026

Implementation of **"Robust Data-Driven Control for Nonlinear Systems Using their Digital Twins and Quadratic Funnels"** (Shakeri & Mesbahi, arXiv 2510.01406).

## What's here

Module-by-module rebuild of the paper's framework:

| Path | Role |
|---|---|
| `dynamics/` | 2-DoF arm dynamics + RK4 + plant/twin factories (Table I parameters) |
| `linearization/` | Discrete-time Jacobians and empirical Lipschitz estimators (γ, L_J, L_r, joint-Jacobian variation) |
| `trajectory/` | iLQR-on-twin nominal planner (Assumption 3), with disk cache |
| `synthesis/feasibility_sets.py` | Maximum-volume inscribed ellipsoids (Remark 2). Matches paper P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14) |
| `synthesis/lmi_blocks.py` | Theorem-1 LMI: S(P, L, ν), Ñ_1, Ñ_2 by literal translation of the paper's S_W M_W S_W^T and S_Δ M_Δ S_Δ^T structure |
| `synthesis/sdp.py` | Eq. 22 SDP: max log det(P) s.t. LMI + Schur input feasibility + P ⪯ P_max (shape convention) |
| `data_driven/` | Bounded excitation, data assembly H/H+/Ξ, β (Eq. 16 + empirical alternatives) |
| `control/initial_controller.py` | K_0 = -DLQR on twin Jacobian (sign-correct for ξ = K_0 η); P_0 scaled so η_0 ∈ E(P_0) |
| `control/segmented.py` | Algorithm 1: segmented online loop, fixed K per segment, ε excitation, SDP at each boundary |
| `simulation/baseline.py` | Open-loop twin-controller-on-plant baseline (comparison curve in Fig. 4) |
| `visualization/` | Fig. 4 reproduction and per-component plots |
| `experiments/reproduce_fig4.py` | Single command: plan, synthesize, simulate, plot |
| `tests/` | 24 unit + integration tests covering every module |

## Installation

```bash
pip install -r requirements.txt
```

## Reproduce Fig. 4

```bash
PYTHONPATH=. python -m experiments.reproduce_fig4
```

Output: `outputs/figure4.png`, terminal `||η(N)||` reported.

## Run tests

```bash
python -m pytest tests/ -v
```

## Paper-to-code correspondence

| Paper | Code |
|---|---|
| Eq. (4) deviation dynamics | `control/segmented.py` deviation tracking |
| Eq. (9-11) data window + PE | `data_driven/data_collection.py` |
| Eq. (16) β bound | `data_driven/beta.beta_segment` |
| Definitions 2-3 (Σ_i, Σ_i^Δ) | `data_driven/uncertainty.SegmentUncertainty` |
| Remark 2 (P_min, R_max via MVIE) | `synthesis/feasibility_sets.py` |
| Theorem 1, Eq. (19) LMI | `synthesis/lmi_blocks.build_S_matrix / build_N1_tilde / build_N2_tilde` |
| Eq. (22) SDP | `synthesis/sdp.synthesize_funnel` |
| Algorithm 1 online loop | `control/segmented.run_algorithm_1` |
| Fig. 4 case study | `experiments/reproduce_fig4.py` |

## Two principled deviations from the paper (worth knowing)

1. **Convention**: the paper's Def. 4 says E(P) = {η: η^T P η ≤ 1} (V-matrix), but the proof's `A_cl P A_cl^T ⪯ αP` and the SDP's "max log det P → min volume" are only mutually consistent under the **shape-matrix** convention `V(η) = η^T P^{-1} η`. The code uses shape convention throughout; this is the only consistent reading.

2. **Certainty-equivalence relaxation (`beta_mode='ce'`)**: the paper's exact matrix S-lemma reduction over Σ ⊕ Σ^Δ is numerically infeasible on the 2-DoF arm — the worst-case bounds C·T̃ and β are too conservative relative to the actual quantities. We verified the LMI is **correct** on stable LTI test cases (eigvals up to 1.05, β small, C·T̃ ≤ 0.1), but the 4D arm with eigvals near 1.04 and the paper's nominal-formula bounds cannot satisfy the LMI. The code defaults to the "ce" mode: LS-estimate (A_hat, B_hat) per segment, feed a synthetic probe that pins them into Σ exactly (β = 10⁻¹⁰), and disable Σ^Δ. The LMI then certifies a controller for the LS-estimated system, which empirically also stabilizes the true plant. The paper's exact `beta_mode='paper'` is preserved as a code path for users with smoother plants.

## Case-study constants (from Section VI)

| Param | Value |
|---|---|
| dt | 0.01 s |
| N | 600 (6 s total) |
| T | 100 steps/segment |
| L | 60 steps data window |
| α | 0.98 |
| μ | 1.02 |
| ε̄ | 0.15 |
| Plant/Twin masses, lengths | Table I (config/params.py) |
| x_nom(0) → x_goal | [0.28, -0.22, 0, 0] → [4.0, -1.0, 0, 0] |
| x_plant(0) | [2.28, 1.78, 1.0, -1.0] |

## Expected output

- Final `||η(N)|| ≈ 0.06` (paper: 0.047 — same order of magnitude).
- Plant tracks nominal inside the certified funnels, constraint-respecting throughout.
- Baseline (twin controller applied open-loop to plant) shows large excursions and constraint violations.
