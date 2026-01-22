"""
Discretization methods for continuous-time dynamics.

Implements RK4 integration for converting continuous-time
dynamics to discrete-time simulation.
"""

from __future__ import annotations

import numpy as np


def rk4_step(f, x, u, dt) -> np.ndarray:
    """Single RK4 integration step."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class DiscreteSystem:
    """Discrete-time wrapper for continuous dynamics."""

    def __init__(self, continuous_dynamics, dt):
        self.continuous_dynamics = continuous_dynamics
        self.dt = dt

    def step(self, x, u) -> np.ndarray:
        return rk4_step(self.continuous_dynamics, x, u, self.dt)

    def simulate(self, x0, u_sequence) -> np.ndarray:
        x = np.asarray(x0, dtype=float)
        trajectory = [x]
        for u in u_sequence:
            x = self.step(x, u)
            trajectory.append(x)
        return np.vstack(trajectory)
