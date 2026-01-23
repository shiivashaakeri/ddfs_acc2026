import numpy as np

from config.params import INITIAL_STATES, LIPSCHITZ, PLANT_PARAMS
from dynamics.digital_twin import compute_mismatch, create_plant, create_twin
from dynamics.discretization import rk4_step
from dynamics.robot_arm import RobotArm2DoF


def test_mass_matrix_positive_definite():
    arm = RobotArm2DoF(**PLANT_PARAMS)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(low=[-np.pi, -np.pi], high=[np.pi, np.pi])
        mass = arm.mass_matrix(q)
        eigvals = np.linalg.eigvals(mass)
        assert np.all(eigvals > 0.0)


def test_rk4_accuracy():
    a = -1.5

    def f(x, u):
        return a * x

    x0 = np.array([1.0])
    dt = 0.05
    x1 = rk4_step(f, x0, np.array([0.0]), dt)
    exact = x0 * np.exp(a * dt)
    assert np.allclose(x1, exact, rtol=1e-7, atol=1e-9)


def test_plant_twin_mismatch_bounded():
    plant = create_plant()
    twin = create_twin()
    x0 = np.array(INITIAL_STATES["x_nom_0"], dtype=float)
    u0 = np.zeros(2, dtype=float)
    mismatch = compute_mismatch(plant, twin, x0, u0)
    assert np.linalg.norm(mismatch) <= 10.0 * LIPSCHITZ["gamma"]

    steps = 100
    u_sequence = np.tile(u0, (steps, 1))
    trajectory = plant.simulate(np.array(INITIAL_STATES["x_plant_0"], dtype=float), u_sequence)
    assert np.all(np.isfinite(trajectory))
    assert np.max(np.abs(trajectory)) < 1e3
    arm = RobotArm2DoF(**PLANT_PARAMS)
    for state in trajectory:
        mass = arm.mass_matrix(state[:2])
        assert np.all(np.linalg.eigvals(mass) > 0.0)


def test_equilibrium_point():
    arm = RobotArm2DoF(**PLANT_PARAMS)
    q = np.array([0.2, -0.1])
    x = np.hstack((q, np.zeros(2)))
    u = arm.gravity_vector(q)
    xdot = arm.dynamics(x, u)
    assert np.allclose(xdot, np.zeros(4), atol=1e-9)
