"""Nominal trajectory planning on the digital twin.

Section VI: "A twin-based LQR plans a nominal from x̂(0) to x_goal with u_goal."
For the 2-DoF arm, the goal is far from the start (~213° shoulder swing) and
the input is bounded, so plain LQR-about-the-goal does not yield a feasible
nominal. We use iterative LQR (iLQR) on the twin dynamics — the standard
nonlinear extension of LQR that produces a locally-optimal feasible nominal
trajectory satisfying Assumption 3.

The returned NominalTrajectory carries both the state sequence (rolled on
the twin) and the open-loop input sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics.digital_twin import create_twin
from dynamics.discretization import DiscreteSystem
from linearization.jacobians import jacobians_discrete


@dataclass
class NominalTrajectory:
    x_nom: np.ndarray  # (N+1, n)
    u_nom: np.ndarray  # (N, m)
    x_goal: np.ndarray
    u_goal: np.ndarray

    @property
    def horizon(self) -> int:
        return self.u_nom.shape[0]

    def increment_bound(self) -> float:
        """v in Assumption 3: max increment of (x̂_nom, û_nom) over k."""
        z = np.hstack([self.x_nom[:-1], self.u_nom])
        return float(np.max(np.linalg.norm(np.diff(z, axis=0), axis=1)))


def _rollout(step, x0, u_seq):
    x = np.zeros((u_seq.shape[0] + 1, x0.size))
    x[0] = x0
    for k in range(u_seq.shape[0]):
        x[k + 1] = step(x[k], u_seq[k])
    return x


def _quadratic_cost(x_seq, u_seq, x_goal, u_goal, Q, R, Qf):
    e_x = x_seq[:-1] - x_goal
    e_u = u_seq - u_goal
    stage = np.einsum("ki,ij,kj->", e_x, Q, e_x) + np.einsum("ki,ij,kj->", e_u, R, e_u)
    terminal = (x_seq[-1] - x_goal) @ Qf @ (x_seq[-1] - x_goal)
    return 0.5 * (stage + terminal)


def ilqr(
    step,
    x0: np.ndarray,
    u_init: np.ndarray,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    u_lo: np.ndarray | None = None,
    u_hi: np.ndarray | None = None,
    max_iter: int = 80,
    tol: float = 1e-6,
    reg_init: float = 1e-3,
    reg_max: float = 1e6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative LQR with Levenberg-style regularization and line search.

    Returns (x_seq, u_seq) of shapes (N+1, n) and (N, m).
    Box constraints on u are enforced by clamping the line-searched update;
    this is the standard simple iLQR-with-box approach.
    """
    n = x0.size
    m = u_init.shape[1]
    N = u_init.shape[0]

    u_seq = u_init.copy()
    x_seq = _rollout(step, x0, u_seq)
    if u_lo is not None and u_hi is not None:
        u_seq = np.clip(u_seq, u_lo, u_hi)
        x_seq = _rollout(step, x0, u_seq)
    J = _quadratic_cost(x_seq, u_seq, x_goal, u_goal, Q, R, Qf)
    reg = reg_init

    for it in range(max_iter):
        # ---- Backward pass: compute feedforward k_seq and feedback K_seq.
        Vx = Qf @ (x_seq[-1] - x_goal)
        Vxx = Qf.copy()
        k_seq = np.zeros((N, m))
        K_seq = np.zeros((N, m, n))
        backward_ok = True
        for t in reversed(range(N)):
            A, B = jacobians_discrete(step, x_seq[t], u_seq[t])
            ex = x_seq[t] - x_goal
            eu = u_seq[t] - u_goal
            Qx = Q @ ex + A.T @ Vx
            Qu = R @ eu + B.T @ Vx
            Qxx = Q + A.T @ Vxx @ A
            Quu = R + B.T @ Vxx @ B + reg * np.eye(m)
            Qux = B.T @ Vxx @ A
            try:
                L_chol = np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                backward_ok = False
                break
            inv_Quu = np.linalg.inv(Quu)
            k_t = -inv_Quu @ Qu
            K_t = -inv_Quu @ Qux
            k_seq[t] = k_t
            K_seq[t] = K_t
            Vx = Qx + K_t.T @ Quu @ k_t + K_t.T @ Qu + Qux.T @ k_t
            Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
            Vxx = 0.5 * (Vxx + Vxx.T)

        if not backward_ok:
            reg = min(reg * 10, reg_max)
            if verbose:
                print(f"iLQR it {it}: backward pass failed, reg -> {reg:.1e}")
            if reg >= reg_max:
                break
            continue

        # ---- Forward pass with line search on the feedforward step.
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            x_new = np.zeros_like(x_seq)
            u_new = np.zeros_like(u_seq)
            x_new[0] = x0
            for t in range(N):
                u_new[t] = u_seq[t] + alpha * k_seq[t] + K_seq[t] @ (x_new[t] - x_seq[t])
                if u_lo is not None:
                    u_new[t] = np.clip(u_new[t], u_lo, u_hi)
                x_new[t + 1] = step(x_new[t], u_new[t])
            J_new = _quadratic_cost(x_new, u_new, x_goal, u_goal, Q, R, Qf)
            if J_new < J:
                if verbose and it % 10 == 0:
                    print(f"iLQR it {it}: J {J:.3e} -> {J_new:.3e} (alpha {alpha})")
                x_seq, u_seq = x_new, u_new
                if abs(J - J_new) < tol * max(1.0, abs(J)):
                    return x_seq, u_seq
                J = J_new
                reg = max(reg / 2, reg_init)
                accepted = True
                break
        if not accepted:
            reg = min(reg * 4, reg_max)
            if verbose:
                print(f"iLQR it {it}: no improvement, reg -> {reg:.1e}")
            if reg >= reg_max:
                break

    return x_seq, u_seq


def plan_nominal_with_twin_lqr(
    x0: np.ndarray,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    N: int,
    twin: DiscreteSystem | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    Qf: np.ndarray | None = None,
    u_lo: np.ndarray | None = None,
    u_hi: np.ndarray | None = None,
    max_iter: int = 80,
    verbose: bool = False,
) -> NominalTrajectory:
    """Plan a nominal trajectory by running iLQR on the twin dynamics."""
    if twin is None:
        twin = create_twin()
    n = x0.size
    m = u_goal.size
    if Q is None:
        Q = 5.0 * np.eye(n)
    if R is None:
        R = 0.5 * np.eye(m)
    if Qf is None:
        Qf = 10.0 * Q

    u_init = np.tile(u_goal, (N, 1))
    x_seq, u_seq = ilqr(
        twin.step, x0, u_init, x_goal, u_goal, Q, R, Qf,
        u_lo=u_lo, u_hi=u_hi, max_iter=max_iter, verbose=verbose,
    )
    return NominalTrajectory(x_nom=x_seq, u_nom=u_seq, x_goal=x_goal, u_goal=u_goal)
