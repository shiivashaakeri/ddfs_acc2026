# Follow-Up Research: What We Found Trying to Fix the Paper

## Summary

We set out to make the paper's exact matrix S-lemma LMI feasible on the 2-DoF arm by introducing a concentration-based β. **The experiment exposed a deeper problem**: the LMI framework itself is fragile in ways that no β tightening can fix. We then tried scenario-based synthesis; it produces non-trivial K but still doesn't stabilize the true plant.

The genuinely working algorithm in this codebase — `ls_lqr` mode — bypasses the LMI for K design entirely, designing K by DLQR on the LS estimate and using the SDP only to certify a funnel.

## Approaches tried, with outcomes

### Approach 1: Tight β with real-data SDP (`prob_real`)

Idea: replace Eq. 16 with `β = κ · ‖W_LS‖²_F` (concentration-motivated).

Result on arm (sweep κ ∈ {1.0, 1.2, 1.5}, P-lower ∈ {None, 0.001 I, 0.01 I, 0.1 I}):

| safety | P_lower | η_final | K norms per segment |
|---|---|---|---|
| 1.0 | None | 625.6 | [3.5, 5.4, 0.2, 0.2, 0.3] (collapse to 0 after seg 1) |
| 1.0 | 0.001 I | 122.6 | [26.3, 9.3, 0.5, 0.3, 0.3] (collapse) |
| 1.0 | 0.1 I | 75.3 | [17.3, 9.6, 0.3, 0.2, 0.2] (collapse) |
| 1.2 | 0.01 I | 140.3 | [24.5, 6.0, 7.1, 7.1, 7.7] (stays bigger but still diverges) |

Even with explicit P-lower-bound, the SDP returns near-trivial K. **Root cause**: real data on the arm has condition number ~127; Σ is elongated; the LMI's max-log-det picks P aligned with the elongated direction, leaving K marginally stabilizing.

### Approach 2: Tight β with synthetic-probe SDP (`prob_synthetic`)

Idea: use well-conditioned synthetic data H=[I, 0], Ξ=[0, I] pinning the LS estimate, with β > 0 representing the LS-residual energy (overfitting-corrected).

Result on arm (sweep κ ∈ {1.0, 1.5, 2.0, 5.0, 100, 1000}):

| safety | η_final | K norms per segment |
|---|---|---|
| 1.0 | 130.6 | [0.01, 0.00, 0.00, 0.00, 0.00] |
| 1.5 | 130.6 | [0.00, 0.00, 0.00, 0.00, 0.00] |
| 2.0 | 130.7 | [0.00, 0.00, 0.00, 0.00, 0.00] |
| 100 | 130.7 | [0.00, 0.00, 0.00, 0.00, 0.00] |

All give degenerate K ≈ 0. Why: well-conditioned data + non-trivial β = "uniform ball around LS estimate" with radius reflecting full residual energy. For the arm, ‖W_LS‖² ≈ 14, which corresponds to perturbations larger than ‖A‖ itself. No K stabilizes the whole ball.

### Approach 3: Scenario-based SDP (`scenario`)

Idea: sample N plants from a Gaussian posterior `(A_s, B_s) ~ N(A_LS, σ² · (ZZ^T)^{-1})`; solve a per-sample Lyapunov LMI for each.

Result on arm (one segment, N ∈ {5, 10, 20, 50}, scale ∈ {0.5, 1.0, 2.0}):

| N | scale | Status | ‖K‖ | |λ(A_cl true)| |
|---|---|---|---|---|
| 5 | 0.5 | optimal_inaccurate | 6.74 | **1.037** (unstable) |
| 10 | 1.0 | optimal_inaccurate | 6.08 | **1.031** |
| 20 | 1.0 | optimal_inaccurate | 5.99 | **1.031** |
| 50 | 1.0 | optimal_inaccurate | 6.01 | **1.031** |

The scenario LMI returns a non-trivial K, but **the true plant is consistently *outside* the convex hull of the posterior samples in a destabilizing direction**. Increasing N doesn't help — the Gaussian posterior's tails don't capture the structured error in the LS estimate.

### Approach 4: `ls_lqr` (separates K-design from funnel-certification)

Designed during the cart-pole investigation. Per segment:
1. LS-estimate (A_LS, B_LS) from data.
2. **K_next = -DLQR(A_LS, B_LS, Q, R)** (no LMI for K).
3. SDP for smallest invariant funnel P given K (LMI is now AFFINE in P alone, no degeneracy).

This is the only approach that works across **all** three case studies:
| System | CE | prob (synth) | scenario | ls_lqr |
|---|---|---|---|---|
| 2-DoF arm | η=0.062 | η=130 | η fails (unstable K) | η ≈ 0.062 |
| Pendulum | η=0.015 | η fails | η fails | η ≈ 0.015 |
| Cart-pole stab | η=10.4 (fail) | η fails | η fails | **η=0.011** |
| Cart-pole swing-up | -- | -- | -- | η=5.8 |

## The actual contribution

The paper's matrix S-lemma is a *joint* synthesis of (P, K): solve for both jointly under data and Jacobian-variation constraints, with the max-log-det objective.

**Our finding**: this joint synthesis is fragile. Three specific failure modes:

1. **Degenerate solution** when β is too big (any non-trivial uncertainty): the SDP returns K ≈ 0 with P ≈ ε I at numerical tolerance, even when the LMI is technically infeasible. The solver's tolerance hides the violation.

2. **Marginal-stability sensitivity**: when the open-loop A has eigenvalues at 1 (integrator modes, common in mechanical systems with position states), the max-log-det objective concentrates P at the feasibility boundary, leaving no margin for K to genuinely stabilize.

3. **Posterior-sample bias**: even with a clean scenario relaxation (no S-procedure), the Gaussian posterior over (A, B) does not capture structured biases in the LS estimate, so the resulting K fails on the true plant.

**The proposed fix**: separate K-design from funnel-certification.

- K via DLQR on LS estimate: avoids the LMI's degenerate fixed points and marginal-stability issues. Doesn't claim formal robustness over Σ; instead claims point-stability for the LS-estimated plant.
- P via the smallest invariant funnel under K: a second SDP with K fixed. This SDP has an affine LMI in P alone, no degeneracy, no max-log-det issue.

Probabilistic robustness can be re-introduced retrospectively: given the designed K, sample plants from the posterior and check that K stabilizes each. If failure rate < δ, we have a probabilistic certificate.

## Suggested paper outline

**Title**: "Decoupled Data-Driven Funnel Control: When the Matrix S-Lemma Fails"

**Sections**:
1. Background and the paper's algorithm.
2. **The three failure modes** (degenerate solution, marginal stability, posterior bias) with reproducible numerical demonstrations.
3. **Decoupled algorithm**: K via certainty-equivalent DLQR + funnel via affine SDP.
4. **Retrospective probabilistic certification**: given K, sample plants, check stability.
5. Empirical comparison on 2-DoF arm, inverted pendulum, cart-pole stabilization & swing-up.
6. Discussion: when does the joint S-lemma LMI work? When must we decouple?

## Recommended next concrete steps for the paper

1. Make the *failure-mode demonstrations* (this repo's `INVESTIGATION.md` + the numerics above) into reproducible figures.
2. Formalize the *decoupled algorithm* (already implemented in `ls_lqr`) with a stability proposition under bounded plant-twin mismatch.
3. Add a *retrospective certification* step: after designing K, randomly sample plants from posterior, run closed-loop sim on each, report empirical stability fraction.
4. Compare to the paper's exact algorithm on the same case studies (which will fail in the documented modes).

This is a more meaningful contribution than "tighter β". It says: the LMI framework is broken in three specific ways, here is exactly how, and here is the working alternative.
