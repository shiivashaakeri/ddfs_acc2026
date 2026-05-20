"""Cart-pole dynamics.

Standard cart-pole formulation: cart of mass M slides horizontally on a
frictionless surface (with viscous cart friction b_c), a pole of mass m
and length l is hinged at the top of the cart (with viscous pivot
friction b_p). State

    x = [p, p_dot, theta, theta_dot]^T

with theta = 0 corresponding to the pole *upright* (unstable equilibrium).
The cart force u acts horizontally on the cart.

Equations of motion (Lagrangian, standard derivation):
    (M + m) p_ddot + m l cos(θ) θ_ddot − m l sin(θ) θ_dot² + b_c p_dot = u
    m l cos(θ) p_ddot + m l² θ_ddot − m g l sin(θ) + b_p θ_dot = 0

We solve the 2x2 mass matrix system at each step.
"""

from __future__ import annotations

import numpy as np


class CartPole:
    def __init__(
        self,
        M: float = 1.0,
        m: float = 0.2,
        l: float = 0.5,
        b_c: float = 0.05,
        b_p: float = 0.005,
        g: float = 9.81,
    ):
        self.M = M
        self.m = m
        self.l = l
        self.b_c = b_c
        self.b_p = b_p
        self.g = g

    def dynamics(self, x, u) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        p, p_dot, theta, theta_dot = x
        force = float(u[0]) if u.ndim else float(u)
        s, c = np.sin(theta), np.cos(theta)
        # Mass matrix
        M11 = self.M + self.m
        M12 = self.m * self.l * c
        M21 = M12
        M22 = self.m * self.l * self.l
        # Right-hand side (mass × accel = rhs)
        rhs1 = force + self.m * self.l * s * theta_dot * theta_dot - self.b_c * p_dot
        rhs2 = self.m * self.g * self.l * s - self.b_p * theta_dot
        # Solve [[M11, M12], [M21, M22]] [p_ddot, theta_ddot]^T = [rhs1, rhs2]^T
        det = M11 * M22 - M12 * M21
        p_ddot = (M22 * rhs1 - M12 * rhs2) / det
        theta_ddot = (-M21 * rhs1 + M11 * rhs2) / det
        return np.array([p_dot, p_ddot, theta_dot, theta_ddot], dtype=float)
