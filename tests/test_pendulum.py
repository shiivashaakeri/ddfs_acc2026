"""Smoke test for the inverted-pendulum case study."""

from __future__ import annotations

import numpy as np

from examples.pendulum.config import (
    INITIAL_STATES,
    PLANT_PARAMS,
    TWIN_PARAMS,
    create_plant,
    create_twin,
)
from examples.pendulum.dynamics import Pendulum
from linearization.jacobians import jacobians_discrete


def test_pendulum_open_loop_stays_at_upright_with_zero_torque():
    plant = create_plant()
    x = np.array([0.0, 0.0])
    for _ in range(50):
        x = plant.step(x, np.zeros(1))
    # Exactly at the unstable equilibrium with no perturbation — stays put.
    assert np.allclose(x, 0.0, atol=1e-8)


def test_pendulum_open_loop_falls_when_perturbed():
    plant = create_plant()
    x = np.array([0.1, 0.0])
    for _ in range(50):
        x = plant.step(x, np.zeros(1))
    # Should have fallen further from upright (no control).
    assert abs(x[0]) > 0.1


def test_pendulum_eigenvalues_match_inverted_pendulum():
    plant = create_plant()
    A, B = jacobians_discrete(plant.step, np.zeros(2), np.zeros(1))
    eigs = np.linalg.eigvals(A)
    # Inverted pendulum: discrete A has one eigenvalue > 1 (unstable mode).
    assert np.any(np.abs(eigs) > 1.0)
    assert np.any(np.abs(eigs) < 1.0)


def test_pendulum_plant_twin_differ():
    plant = create_plant(); twin = create_twin()
    x = np.array([0.3, 0.5])
    u = np.array([0.5])
    assert not np.allclose(plant.step(x, u), twin.step(x, u))
    # Mismatch parameters check
    assert PLANT_PARAMS["m"] != TWIN_PARAMS["m"]
    assert PLANT_PARAMS["l"] != TWIN_PARAMS["l"]


def test_pendulum_dynamics_class_consistency():
    pend = Pendulum(m=1.0, l=1.0, b=0.0, g=9.81)
    # Hanging straight down (θ=π) and at rest with no torque: should stay still
    # (down is the stable equilibrium for our sign convention where θ=0 is upright).
    x = np.array([np.pi, 0.0])
    xd = pend.dynamics(x, np.array([0.0]))
    # Gravity term at θ=π: (g/l) * sin(π) = 0. So acceleration is zero.
    assert abs(xd[1]) < 1e-10
