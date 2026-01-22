"""
2-DoF Robot Arm Dynamics.

Implements the continuous-time dynamics for a planar 2-DoF robot arm
with standard rigid-body dynamics (Spong et al., 2006).

State: x = [q1, q2, q1_dot, q2_dot]^T (4-dim)
Input: u = [tau1, tau2]^T (2-dim)

Key equations:
    x_dot = [q_dot; M(q)^{-1}(tau - C(q,q_dot)q_dot - G(q) - B*q_dot)]

    M(q) = [[a+2b*cos(q2), d+b*cos(q2)], [d+b*cos(q2), d]]
    where:
        a = I1 + I2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
        b = m2*l1*lc2
        d = I2 + m2*lc2^2
"""

from __future__ import annotations

import numpy as np


class RobotArm2DoF:
    """Continuous-time planar 2-DoF robot arm dynamics."""

    def __init__(self, m1, m2, l1, l2, lc1, lc2, I1, I2, b1, b2, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.lc1 = lc1
        self.lc2 = lc2
        self.I1 = I1
        self.I2 = I2
        self.b1 = b1
        self.b2 = b2
        self.g = g

    def _inertia_terms(self) -> tuple[float, float, float]:
        a = (
            self.I1
            + self.I2
            + self.m1 * self.lc1**2
            + self.m2 * (self.l1**2 + self.lc2**2)
        )
        b = self.m2 * self.l1 * self.lc2
        d = self.I2 + self.m2 * self.lc2**2
        return a, b, d

    def mass_matrix(self, q) -> np.ndarray:
        q1, q2 = q
        a, b, d = self._inertia_terms()
        cos_q2 = np.cos(q2)
        return np.array(
            [[a + 2.0 * b * cos_q2, d + b * cos_q2], [d + b * cos_q2, d]],
            dtype=float,
        )

    def coriolis_matrix(self, q, qdot) -> np.ndarray:
        _, q2 = q
        q1_dot, q2_dot = qdot
        b = self._inertia_terms()[1]
        sin_q2 = np.sin(q2)
        return np.array(
            [
                [-2.0 * b * sin_q2 * q2_dot, -b * sin_q2 * q2_dot],
                [b * sin_q2 * q1_dot, 0.0],
            ],
            dtype=float,
        )

    def gravity_vector(self, q) -> np.ndarray:
        q1, q2 = q
        g = self.g
        term1 = (self.m1 * self.lc1 + self.m2 * self.l1) * g * np.cos(q1)
        term2 = self.m2 * self.lc2 * g * np.cos(q1 + q2)
        return np.array([term1 + term2, term2], dtype=float)

    def friction_matrix(self) -> np.ndarray:
        return np.array([[self.b1, 0.0], [0.0, self.b2]], dtype=float)

    def dynamics(self, x, u) -> np.ndarray:
        q = np.asarray(x[:2], dtype=float)
        qdot = np.asarray(x[2:], dtype=float)
        tau = np.asarray(u, dtype=float)
        mass = self.mass_matrix(q)
        coriolis = self.coriolis_matrix(q, qdot)
        gravity = self.gravity_vector(q)
        friction = self.friction_matrix()
        qddot = np.linalg.solve(mass, tau - coriolis @ qdot - gravity - friction @ qdot)
        return np.hstack((qdot, qddot))
