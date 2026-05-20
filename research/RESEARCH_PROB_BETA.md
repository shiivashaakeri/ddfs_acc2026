# Probabilistic β: Concentration-Based Disturbance Bound for the Funnel LMI

## Problem

The paper's β (Eq. 16) is the sum of squared **pointwise worst-case** bounds from Lemma 3:

$$\beta_i^{\text{paper}} = \sum_{k \in \mathcal{T}_i^D} \left( C |k - k_i| \|z(k)\| + \gamma + L_r \|z(k)\|^2 \right)^2$$

It is universally valid but **wildly over-conservative**: measured on the paper's own 2-DoF arm case study, $\beta_i^{\text{paper}}$ overestimates $\|W_i\|_F^2$ by **5–7 orders of magnitude** (see `INVESTIGATION.md` §2).

Consequence: the data-consistent set $\Sigma_i = \{(A, B): \|H_i^+ - A H_i - B \Xi_i\|_F^2 \leq \beta_i\}$ becomes so large that the matrix S-lemma's LMI is either infeasible or admits a degenerate solution ($K \approx 0$, $P \to 0$). The "certified" controller does not actually stabilize the plant.

## Proposed bound

Replace the worst-case β with a **probabilistic** one derived from the LS-fit residual.

### Setup

For data window $\mathcal{T}_i^D$ of length $L$, stack
$$Z_i = \begin{bmatrix} H_i \\ \Xi_i \end{bmatrix} \in \mathbb{R}^{(n+m) \times L},\qquad H_i^+ \in \mathbb{R}^{n \times L}.$$

The least-squares estimate is
$$[\hat{A}_i, \hat{B}_i] = H_i^+ Z_i^\top (Z_i Z_i^\top)^{-1}.$$

Its residual matrix is
$$W_i^{\text{LS}} = H_i^+ - [\hat{A}_i, \hat{B}_i] Z_i = H_i^+ \Pi_{Z_i}^\perp$$

where $\Pi_{Z_i}^\perp = I - Z_i^\top (Z_i Z_i^\top)^{-1} Z_i$ is the projection onto the row-null-space of $Z_i$.

### Key relation

The true plant satisfies $H_i^+ = A_i H_i + B_i \Xi_i + W_i$ with the *true* disturbance $W_i$. By the Pythagorean identity for the LS projection,
$$\|W_i^{\text{LS}}\|_F^2 = \|W_i \Pi_{Z_i}^\perp\|_F^2 \leq \|W_i\|_F^2.$$

Equality fails when the disturbance has a component "in the data direction" $Z_i^\top$.

### Concentration bound on $\|W_i\|_F^2 / \|W_i^{\text{LS}}\|_F^2$

**Assumption (mild).** The aggregated disturbance $W_i = [w_i(k_i^D), \ldots, w_i(k_{i+1}-1)]$, viewed as a random matrix conditional on the trajectory, has entries that are bounded in magnitude by the Lemma-3 envelope $\bar w_k = C|k-k_i| \|z(k)\| + \gamma + L_r \|z(k)\|^2$, and the entries across different time steps are *weakly correlated*.

Under this assumption, $\|W_i\|_F^2 = \sum_k \|w_i(k)\|^2$ is a sum of $L$ bounded sub-quantities. Bernstein's inequality gives:
$$\Pr\left(\|W_i\|_F^2 > \mathbb{E}\|W_i\|_F^2 + t\right) \leq \exp\left(-\frac{c t^2}{n L \bar w_{\max}^4 + \bar w_{\max}^2 t}\right)$$

for some absolute constant $c$.

The expected energy $\mathbb{E}\|W_i\|_F^2$ is captured (up to a known overfitting factor) by the LS residual $\|W_i^{\text{LS}}\|_F^2$. For Gaussian disturbances with variance $\sigma^2$ per entry, $\|W_i^{\text{LS}}\|_F^2 \sim \sigma^2 \chi^2_{n(L - (n+m))}$ and $\|W_i\|_F^2 \sim \sigma^2 \chi^2_{nL}$, so
$$\frac{\mathbb{E} \|W_i\|_F^2}{\mathbb{E} \|W_i^{\text{LS}}\|_F^2} = \frac{nL}{n(L - (n+m))} = \frac{L}{L - (n+m)}.$$

For typical parameters ($L = 60$, $n + m = 6$): the ratio is $60/54 \approx 1.11$.

### Practical β

Combining the LS-overfitting correction with a Bernstein-style safety margin for sub-Gaussian tails at confidence $1 - \delta$:
$$\boxed{\beta_i^{\text{prob}} = \kappa_\delta \cdot \frac{L}{L - (n+m)} \cdot \|W_i^{\text{LS}}\|_F^2}$$

with $\kappa_\delta = 1 + \mathcal{O}(\sqrt{(1/L) \log(1/\delta)})$. For $\delta = 0.05$ and $L = 60$: $\kappa_\delta \approx 1 + 0.22 = 1.22$.

So $\beta^{\text{prob}} \approx 1.35 \cdot \|W_i^{\text{LS}}\|_F^2$ — only ~35% bigger than the LS residual, vs paper's $10^5\times$ bigger.

## Anti-degeneracy: explicit P lower bound

A tight β is necessary but **not sufficient** — `INVESTIGATION.md` §4 shows the SDP can still return $K \approx 0$ with $P \to \varepsilon I$ at numerical tolerance. Block this with:

$$P \succeq P_{\text{lower}}$$

where $P_{\text{lower}} = \rho I$ for some $\rho > 0$ (e.g., $\rho = \lambda_{\min}(P_{\max})/100$). This forces a non-degenerate funnel and pushes the SDP to find a genuine non-zero $K$ — or report INFEASIBLE if no such $K$ exists for $\Sigma_i$ (which is the *honest* answer).

## Putting it together

The new `prob` mode in `control/segmented.py`:

1. At each segment boundary, compute the LS residual $W_i^{\text{LS}}$ from the data.
2. Set $\beta_i^{\text{prob}} = 1.35 \cdot \|W_i^{\text{LS}}\|_F^2$ (or generalized concentration constant).
3. Call `synthesize_funnel(..., P_lower_shape=P_lower)` with the explicit P-lower bound.
4. The SDP returns either a genuinely stabilizing $(P_{i+1}, K_{i+1})$ certified for $\Sigma_i$, or reports infeasible (signaling that the data window is genuinely uninformative).

## Theoretical claim

**Proposition (informal).** Under bounded excitation $\|\epsilon(k)\| \leq \bar\epsilon$, smoothness of the dynamics (Lipschitz Jacobian), and the persistence-of-excitation rank condition, the probabilistic β $\beta^{\text{prob}}$ contains the true $\|W_i\|_F^2$ with probability at least $1 - \delta$. Consequently, the LMI `synthesize_funnel(β = β^{prob}, P_lower)` returns a controller that is **certified for the true plant** with the same probability.

The formal proof would adapt Hanson-Wright + LS overfitting theory; a credible-set version of this proposition exists in [Dean et al., 2017] for LQR-style problems.

## Empirical validation — and a sharper negative finding

We implemented the probabilistic β + P-lower-bound and tested on the arm case study. **The fix does NOT work as expected.** Across safety factors κ ∈ {1.0, 1.5, 2.0, 5.0, 100, 1000}, the SDP returns K ≈ 0 (magnitudes < 10⁻²) and the closed loop diverges to ‖η(N)‖ ≈ 130.

This is a **stronger negative result** than expected. It says the matrix S-lemma's robustness mechanism doesn't just have a "conservatism problem" with bad β — it has a **structural problem with any non-trivial β**.

### Why the fix fails

Two attempted configurations:

**(i) Real data H, H⁺, Ξ + small β (κ × ‖W_LS‖²) + P-lower bound.** The real data has condition number ~127 on the arm; the resulting Σ is heavily elongated in poorly-excited directions. The LMI's max-log-det objective picks K to satisfy the worst-case direction; that direction is precisely the poorly-identified one, so K barely stabilizes anywhere.

**(ii) Synthetic well-conditioned data H = [I, 0], Ξ = [0, I] + non-zero β.** Now Σ is a clean ball in (A, B) space around the LS estimate, but the *radius* (β) reflects total LS residual energy ‖W_LS‖² which on real systems is non-trivial (≈ 14 for the arm). The "ball" then contains plants whose A perturbations have spectral norm up to √14 ≈ 4 — larger than ‖A‖ itself. No K can stabilize that whole set; SDP returns K = 0 trivially.

### The actual root cause

The matrix S-lemma's reduction from the implication

$$\forall (A, B) \in \Sigma_i: \alpha P − A_{\text{cl}} P A_{\text{cl}}^\top \succeq 0$$

to the LMI

$$M(P, K) − \lambda_1 N_1 − \lambda_2 N_2 \succeq 0$$

is an **inner approximation** (sufficient but not necessary). For non-trivial Σ, the gap between the two is wide; even when a stabilizing K exists for every plant in Σ, the LMI may admit only K = 0.

This is fundamental — it cannot be patched by adjusting β or adding constraints to the SDP.

## Revised proposal: scenario-based synthesis

The S-procedure's lossiness is the real bottleneck. A better approach:

1. **Sample N plants from Σ_i** (e.g., draw $(A_s, B_s)$ from a Bayesian posterior with mean = LS estimate, covariance from data Gram + LS residual).
2. **Solve a per-sample LMI**: find $P, L$ such that the Lyapunov LMI holds for EACH sampled plant individually:
$$\forall s: \begin{bmatrix} \alpha P & A_s P + B_s L \\ \star & P \end{bmatrix} \succeq 0$$
3. With sufficient $N$, the resulting $(P, K)$ stabilises all of Σ_i with high probability (Calafiore-Campi sample bound on the scenario approach).

This is a **tighter relaxation** than the S-lemma: no Lagrangian multipliers $\lambda_i$, no inner approximation. The per-sample LMI is feasible iff a uniform stabilizer exists.

We already implemented this in `INVESTIGATION.md` §3's diagnostic (10 random plants); it gave clean K with worst-case |λ(A_cl)| = 0.92 even when the matrix S-lemma returned K = 0. **The contribution is to make this the production algorithm**, not just a diagnostic.

## Updated theoretical claim

**Proposition (informal).** Under bounded excitation, smoothness of the dynamics, and N samples from a posterior over (A, B), the scenario LMI

$$\text{find } P \succ 0, L: \forall s = 1, \ldots, N: \begin{bmatrix} \alpha P & A_s P + B_s L \\ \star & P \end{bmatrix} \succeq 0$$

is feasible iff a stabilizer exists for the convex hull of $\{(A_s, B_s)\}$. By the Calafiore-Campi bound, with $N = O((n(n+m) + \log(1/\delta))/\varepsilon)$ samples, the resulting $(P, K)$ stabilises a fraction $1 - \varepsilon$ of the posterior support with probability $1 - \delta$.

This is the principled replacement for the matrix S-lemma.

## Next step

Implement scenario-based synthesis as a new mode (`beta_mode="scenario"`) and test against the same three case studies. **Expected result**: works on arm + pendulum + cart-pole stabilization without requiring `ls_lqr` workaround, and with formal probabilistic guarantees.
