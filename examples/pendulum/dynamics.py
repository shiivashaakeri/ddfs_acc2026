"""Inverted pendulum dynamics.

Single rigid link with point mass m at distance l from a frictional pivot,
gravity g acting down. State x = [θ, θ̇]^T where θ = 0 is the *upright*
posture (unstable equilibrium). Input u = τ (torque at pivot).

Continuous dynamics:
    I θ̈ = m g l sin(θ) − b θ̇ + τ,    I = m l²

so
    θ̈ = (g/l) sin(θ) − (b / (m l²)) θ̇ + τ / (m l²).

The (θ=0, θ̇=0) point is unstable open-loop: linearising gives
    A_c = [[0, 1], [g/l, −b/(m l²)]],  B_c = [[0], [1/(m l²)]].
"""

from __future__ import annotations

import numpy as np


class Pendulum:
    """Continuous-time inverted-pendulum dynamics."""

    def __init__(self, m: float, l: float, b: float, g: float = 9.81):
        self.m = m
        self.l = l
        self.b = b
        self.g = g

    @property
    def I(self) -> float:
        return self.m * self.l * self.l

    def dynamics(self, x, u) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        theta, theta_dot = x[0], x[1]
        tau = u[0] if u.ndim else float(u)
        theta_ddot = (self.g / self.l) * np.sin(theta) - (self.b / self.I) * theta_dot + tau / self.I
        return np.array([theta_dot, theta_ddot], dtype=float)
