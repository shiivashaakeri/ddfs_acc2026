"""Estimate the constants γ, L_J, L_r used in the disturbance bounds.

- γ (Assumption 1): uniform upper bound on one-step plant–twin mismatch
  ||Δ(x,u)|| = ||f_d^plant(x,u) − f_d^twin(x,u)||.

- L_J (Lemma 2): Lipschitz constant of the joint Jacobian map
  (x,u) ↦ [A(x,u)  B(x,u)] on X×U, in spectral norm.

- L_r (Lemma 1): second-order remainder constant satisfying
  ||f_d(x+η, u+ξ) − f_d(x,u) − A(x,u)η − B(x,u)ξ|| ≤ L_r ||[η;ξ]||².

All three are estimated by sampling along the nominal trajectory and taking
the empirical max — this is exactly how Section VI quotes its numbers
(γ = 0.034, L_J = 16.94, L_r = 0.074 for the 2-DoF arm case study).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from linearization.jacobians import jacobians_discrete


def estimate_gamma(
    plant_step: Callable, twin_step: Callable, x_nom: np.ndarray, u_nom: np.ndarray
) -> float:
    """Max one-step mismatch along the nominal trajectory."""
    N = u_nom.shape[0]
    return float(
        max(
            np.linalg.norm(plant_step(x_nom[k], u_nom[k]) - twin_step(x_nom[k], u_nom[k]))
            for k in range(N)
        )
    )


def estimate_L_J(
    plant_step: Callable,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    radius: float = 0.05,
    n_probes: int = 6,
    seed: int = 0,
) -> float:
    """Lipschitz constant of the joint Jacobian on a neighbourhood of the nominal.

    Probes nearby (x, u) pairs and measures ||[ΔA  ΔB]||₂ / ||[Δx; Δu]||.
    """
    N = u_nom.shape[0]
    rng = np.random.default_rng(seed)
    L = 0.0
    for k in range(N):
        x0, u0 = x_nom[k], u_nom[k]
        A0, B0 = jacobians_discrete(plant_step, x0, u0)
        for _ in range(n_probes):
            dx = rng.normal(size=x0.size)
            du = rng.normal(size=u0.size)
            scale = radius / np.linalg.norm(np.concatenate([dx, du]))
            dx, du = dx * scale, du * scale
            A1, B1 = jacobians_discrete(plant_step, x0 + dx, u0 + du)
            diff = np.hstack([A1 - A0, B1 - B0])
            denom = np.linalg.norm(np.concatenate([dx, du]))
            if denom > 0:
                L = max(L, np.linalg.norm(diff, 2) / denom)
    return float(L)


def estimate_L_r(
    plant_step: Callable,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    radius: float = 0.05,
    n_probes: int = 8,
    seed: int = 1,
) -> float:
    """Second-order remainder constant for the discrete plant map.

    L_r = sup ||f_d(x+η,u+ξ) - f_d(x,u) - Aη - Bξ|| / ||[η;ξ]||².
    """
    N = u_nom.shape[0]
    rng = np.random.default_rng(seed)
    L = 0.0
    for k in range(N):
        x0, u0 = x_nom[k], u_nom[k]
        A0, B0 = jacobians_discrete(plant_step, x0, u0)
        f0 = plant_step(x0, u0)
        for _ in range(n_probes):
            dx = rng.normal(size=x0.size)
            du = rng.normal(size=u0.size)
            z_norm = np.linalg.norm(np.concatenate([dx, du]))
            scale = radius / z_norm
            dx, du = dx * scale, du * scale
            z_norm = radius
            f1 = plant_step(x0 + dx, u0 + du)
            lin = f0 + A0 @ dx + B0 @ du
            L = max(L, np.linalg.norm(f1 - lin) / (z_norm ** 2))
    return float(L)
