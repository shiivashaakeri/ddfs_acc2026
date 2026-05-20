# Why the Paper's Theory Doesn't Work in Practice

## TL;DR

The paper's theory is **internally consistent and mathematically correct**, but it stacks **four sources of conservatism** that together make the algorithm fail on any realistic plant:

1. **β bound (Lemma 3 / Eq. 16) overestimates `||W||_F²` by 10⁵–10⁷×**
2. **Matrix S-procedure with two QMIs is only an inner approximation**, lossy when the uncertainty set is large
3. **The LMI admits a non-physical degenerate solution** (K≈0 with P≈εI) that the solver returns when the genuine LMI is infeasible
4. **The "max log det P" objective fails for plants with marginal-stability modes** (eigenvalues near 1) even with zero uncertainty

Below: each issue with numerical evidence from this repo.

---

## 1. The LMI itself is mathematically sound

Five algebraic identity tests in `tests/test_lmi_blocks.py` verify the paper's construction line-by-line:

- `Φ^T M Φ = α P − A_cl P A_cl^T` (the original QMI from the proof)
- `Φ^T N₁ Φ = β I − W Wᵀ` (data-consistency)
- `Φ^T N₂ Φ = C²T̃² I − ΔA ΔAᵀ − ΔB ΔBᵀ` (Jacobian variation)
- Schur padding: `Schur((6,6)=P) on S = M with L=KP`
- LTI test bed: on a stable 2D LTI with eigvals up to 1.05, the SDP solves cleanly and returns a stabilizing K.

The LMI construction is fine. The trouble is upstream.

---

## 2. Issue #1 — β from Eq. (16) is wildly conservative

Eq. (16) sums squared *pointwise worst-case* bounds from Lemma 3:

```
β_i = Σ_{k ∈ T_i^D}  ( C |k − k_i| ||z(k)|| + γ + L_r ||z(k)||² )²
```

This treats every step as worst-case: it assumes `A(k) − A(s)` simultaneously achieves the maximum spectral norm bound AND aligns with the worst direction of `z(k)`. That's never tight.

**Measured on the arm case study (paper parameters):**

| Segment | β from Eq. 16 | ‖W‖_F² (plant Jac at seg start) | ‖W‖_F² (LS-best-fit) | β / ‖W‖² |
|:-:|:-:|:-:|:-:|:-:|
| 0 | 1.11×10⁹ | 1.5×10³ | 1.4×10¹ | 7.3×10⁵ |
| 1 | 1.73×10⁹ | 1.0×10³ | 2.4×10² | 1.7×10⁶ |
| 2 | 5.08×10⁹ | 7.5×10³ | 1.1×10³ | 6.7×10⁵ |
| 3 | 1.28×10¹⁰ | 1.9×10⁴ | 1.2×10³ | 6.9×10⁵ |
| 4 | 2.97×10¹⁰ | 4.1×10⁴ | 1.0×10³ | 7.3×10⁵ |
| 5 | 6.32×10¹⁰ | 8.3×10⁴ | 6.6×10² | 7.7×10⁵ |

The paper's `β` is **10⁵–10⁷ times larger** than the actual disturbance energy. Why this matters: the set `Σ_i = {(A, B) : ‖H⁺ − AH − BΞ‖² ≤ β}` then contains an enormous family of "plants", many of which are not even stabilizable by a fixed K. The matrix S-lemma must produce a controller robust to ALL of them — which often doesn't exist.

The same blow-up happens on the inverted pendulum: `β` is in the 10⁴ range when actual is < 10⁻³.

---

## 3. Issue #2 — The matrix S-procedure with two QMIs is lossy

For ONE quadratic constraint Yakubovich's S-lemma is **exact** (lossless):
```
∀ Φ with Φᵀ N Φ ⪰ 0  ⇒  Φᵀ M Φ ⪰ 0
⇔
∃ λ ≥ 0 : M − λN ⪰ 0
```

For TWO quadratic constraints (`N₁` for data + `N₂` for Jacobian variation) the LMI form `∃ λ₁, λ₂ ≥ 0 : M − λ₁N₁ − λ₂N₂ ⪰ 0` is only **sufficient** for the implication, not necessary. The relaxation can be empty even when a controller exists.

**Direct demonstration** (2D LTI, 10 random perturbed plants in a small uncertainty set):

| Method | Status | K | Worst |λ(A_cl)| |
|:-:|:-:|:-:|:-:|
| Per-sample robust LMI (sees each plant directly) | optimal | (−3.89, −1.78) | 0.92 |
| Matrix S-lemma, β = tight (0.0031) | optimal | (−2.00, −0.79) | 0.89 |
| Matrix S-lemma, β = 0.1 (30× too big) | "optimal" | (−2.2e-4, −4.1e-4) ≈ 0 | **1.06 ← UNSTABLE** |
| Matrix S-lemma, β = 1.0 | "optimal" | ≈ 0 | **1.06 ← UNSTABLE** |

When β is even modestly inflated past its tight value, the S-lemma quietly accepts a degenerate "solution" with `K ≈ 0`.

---

## 4. Issue #3 — The SDP's degenerate solution at large β

When the genuine LMI is infeasible, **the solver returns a near-feasible solution** at numerical tolerance. Manually checking the actual Lyapunov margin exposes the violation.

**Same 2D problem, sweeping β:**

| β | log det P | ‖K‖ | |λ(A_cl)| | min eig(αP − A_cl P A_cl^T) | min eig(P) |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.001 | +32.6 | 11.4 | 0.46 | **+6.8×10⁵** ✓ | 7.9×10⁵ |
| 0.005 | −12.3 | 2.1 | 0.91 | +1.1×10⁻⁵ ✓ | 5.4×10⁻⁴ |
| 0.01 | −14.2 | 0.20 | **1.10** | **−1.4×10⁻⁴** ✗ | 4.2×10⁻⁴ |
| 0.05 | −16.0 | 0.13 | 1.10 | −1.5×10⁻⁴ ✗ | 7.1×10⁻⁵ |
| 0.1 | −15.9 | 6.6×10⁻⁴ | 1.02 | −1.2×10⁻⁴ ✗ | 7.7×10⁻⁵ |
| 1.0 | −19.4 | 0.52 | 1.02 | −5.9×10⁻⁶ ✗ | 5.9×10⁻⁵ |

The Lyapunov margin's minimum eigenvalue **goes negative** at β = 0.01 — meaning the LMI is *technically infeasible* but CLARABEL/SCS report "optimal" because the violation (10⁻⁴) is within their numerical tolerance. The "solution" has `K` so small it doesn't stabilize the true plant.

This is the mechanism behind the K≈0 trivial solutions we observed on the arm.

---

## 5. Issue #4 — `max log det P` fails for marginal-stability plants

Even with **zero uncertainty** (β = 10⁻¹⁰, exact data, no Jacobian variation), the paper's "maximize log det P" objective gives an unstabilizing K when the open-loop has eigenvalues near the unit circle.

**Cart-pole at the upright equilibrium** (A eigvals: 1, 1, 1.05, 0.95):

| Objective | ‖K‖ | |λ(A_cl)| on true plant | Notes |
|:-:|:-:|:-:|:-:|
| `max log det P` (paper) | 91.7 | **1.020 ← UNSTABLE** | P at the P_max boundary |
| `min trace P` | 920.1 | **1.064 ← UNSTABLE** | P collapses toward 0 |
| `DLQR(A, B, Q, R)` (our `ls_lqr` mode) | 57.1 | **0.979** ✓ STABLE | Bypasses SDP entirely |

Both SDP objectives return unstabilizing K. Only direct DLQR works.

**Why this happens**: with marginal eigenvalues, the Lyapunov LMI `A_cl P A_clᵀ ⪯ α P` is barely feasible — any P satisfying it lies on the boundary of the feasible region. The SDP's objective (whether maximize or minimize) pulls P further to the boundary; the resulting K stabilizes only at numerical tolerance and fails on the nonlinear plant.

---

## 6. Issue #5 (cosmetic) — Notation inconsistency in the paper

- **Def. 4** writes `E(P) = {η : ηᵀ P η ≤ 1}` (V-matrix convention; invariance requires `Aᵀ P A ⪯ P`).
- **Proof of Theorem 1** writes `α P − A_cl P A_clᵀ ⪰ 0` (shape-matrix convention; invariance requires `A P Aᵀ ⪯ P`).

These two conventions are equivalent under the substitution `P ↔ P⁻¹`, but the paper uses Def. 4 notation in the prose and shape-matrix algebra in the proofs. We resolved it by using the **shape-matrix convention** throughout (the only reading under which the proof's LMI, the SDP objective `max log det P → min volume`, the input feasibility `R = K P⁻¹ Kᵀ`, and Lemma 5's `P ⪰ P_min` are *mutually consistent*).

This is a typo, not a math error. But it cost us a half-day of debugging.

---

## 7. Summary of root causes

| Issue | Mechanism | Fix |
|:--|:--|:--|
| β overestimates ‖W‖² by 10⁵–10⁷× | Sum of squared pointwise worst cases | Empirical β from LS-fit residual, or CE relaxation |
| S-procedure with two QMIs is lossy | Inner approximation | Treat Σ₁ ⊕ Σ₂ as a single SDP with tight β |
| SDP returns K≈0 at numerical tolerance when LMI is infeasible | Solver accepts near-feasibility | Verify post-hoc: actually compute Lyapunov margin |
| `max log det P` fails for marginal stability | Degenerate P at feasibility boundary | DLQR-on-LS-estimate (our `ls_lqr` mode) |
| Notation inconsistency | Paper has typo | Use shape-matrix convention consistently |

---

## 8. Why our three case studies behave the way they do

- **2-DoF arm (CE mode, η_final = 0.062)**: works because CE mode replaces the paper-conservative β with empirical β ≈ 0; the SDP's max-log-det then returns a non-trivial K. The remaining gap (paper reports 0.047) is the inflation factor — our SDP-certified funnel doesn't actually contain the plant's transient.

- **Inverted pendulum (CE mode, η_final = 0.015)**: same story; smaller dimension makes everything cleaner.

- **Cart-pole stabilization (CE diverges, ls_lqr works with η_final = 0.011)**: CE fails because of issue #4 (max-log-det on marginal stability). `ls_lqr` bypasses by computing K via DLQR-on-LS-estimate.

- **Cart-pole swing-up (η_final = 5.8)**: even `ls_lqr` struggles because the catch phase has rapidly-varying Jacobians — the LS-estimate from the last segment isn't valid for the new operating point. A truly adaptive controller would re-estimate within a segment.

---

## 9. What would actually fix the paper

In rough order of impact:

1. **Replace Eq. 16's β with a probabilistic / data-tight bound.** E.g., β = (1 + ε) · ‖W_LS‖_F² where W_LS is the LS residual; this captures the actual noise level rather than the worst-case Lipschitz bound.

2. **Add an upper bound on P** explicitly (`P ⪰ P_lower > 0`) to prevent the degenerate `P → 0, K → 0` collapse. The SDP must then find a non-trivial controller.

3. **Switch the K-design** from "solve LMI for joint (P, K)" to "solve DLQR-on-LS-estimate for K, then LMI for P alone given K". This is what our `ls_lqr` mode does.

4. **Replace the matrix S-lemma's pointwise relaxation** with a tighter scenario-based approach (sample plants from Σ, solve a per-sample LMI).

5. **Fix the notation** to be consistent (use shape-matrix everywhere, or V-matrix everywhere — not mixed).

The paper's structural framework (segmented online learning + funnel synthesis + LMI certificates) is sound. The specific bound and objective choices are what fail.
