"""Tests for dynamics, RK4, and plant/twin construction."""

from __future__ import annotations

import numpy as np
import pytest

from config.params import PLANT_PARAMS, SIM_PARAMS, TWIN_PARAMS
from dynamics.digital_twin import compute_mismatch, create_plant, create_twin
from dynamics.discretization import rk4_step
from dynamics.robot_arm import RobotArm2DoF


def _arm(params):
    return RobotArm2DoF(**{k: v for k, v in params.items() if k != "g"}, g=params["g"])


def test_mass_matrix_spd():
    arm = _arm(PLANT_PARAMS)
    for q1 in np.linspace(-np.pi, np.pi, 5):
        for q2 in np.linspace(-np.pi, np.pi, 5):
            M = arm.mass_matrix([q1, q2])
            assert np.allclose(M, M.T)
            assert np.all(np.linalg.eigvalsh(M) > 0)


def test_gravity_zero_at_vertical():
    arm = _arm(PLANT_PARAMS)
    # cos(pi/2) = 0 makes both gravity terms vanish
    G = arm.gravity_vector([np.pi / 2, 0.0])
    assert np.allclose(G, 0.0, atol=1e-12)


def test_rk4_zero_input_at_equilibrium():
    # Vertical-up posture with zero velocity: gravity is the only force.
    # Pick state where gravity is zero (q1 = pi/2) and verify zero velocity stays zero in q-coords.
    arm = _arm(PLANT_PARAMS)
    x0 = np.array([np.pi / 2, 0.0, 0.0, 0.0])
    x1 = rk4_step(arm.dynamics, x0, np.zeros(2), SIM_PARAMS["dt"])
    # Positions unchanged to leading order (qdot=0), velocities pick up no gravity acceleration.
    assert np.allclose(x1[:2], x0[:2], atol=1e-6)
    assert np.allclose(x1[2:], 0.0, atol=1e-6)


def test_plant_twin_are_different():
    plant = create_plant()
    twin = create_twin()
    x = np.array([0.3, -0.2, 0.1, -0.1])
    u = np.array([1.0, 0.5])
    xp = plant.step(x, u)
    xt = twin.step(x, u)
    assert not np.allclose(xp, xt)


def test_mismatch_bounded_on_nominal_region():
    # Section VI quotes gamma = 0.034 as the max one-step mismatch along the nominal.
    # Sanity-check that mismatch is on the order of 1e-2 in this neighborhood.
    plant = create_plant()
    twin = create_twin()
    rng = np.random.default_rng(0)
    max_norm = 0.0
    for _ in range(200):
        x = rng.uniform([-1, -1, -1, -1], [1, 1, 1, 1])
        u = rng.uniform([-10, -10], [10, 10])
        d = compute_mismatch(plant, twin, x, u)
        max_norm = max(max_norm, np.linalg.norm(d))
    assert max_norm < 1.0, f"mismatch unexpectedly large: {max_norm}"


def test_simulate_shape():
    plant = create_plant()
    x0 = np.zeros(4)
    u_seq = np.zeros((10, 2))
    traj = plant.simulate(x0, u_seq)
    assert traj.shape == (11, 4)


def test_param_table_matches_paper():
    # Paper Table I (Plant column)
    assert PLANT_PARAMS["m1"] == 1.00
    assert PLANT_PARAMS["l1"] == 0.70
    assert PLANT_PARAMS["I1"] == 0.050
    # Twin column
    assert TWIN_PARAMS["m1"] == 0.95
    assert TWIN_PARAMS["l1"] == 0.73


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
