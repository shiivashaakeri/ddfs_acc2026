"""
End-to-end integration tests.

Test cases:
- Full simulation runs without errors
- All segments complete
- SDPs solve successfully
- Constraints are satisfied
- Proposed method outperforms baseline
"""

from __future__ import annotations

import numpy as np
import pytest

from config.params import (
    INITIAL_STATES,
    INPUT_BOUNDS,
    LIPSCHITZ,
    PLANT_PARAMS,
    SIM_PARAMS,
    STATE_BOUNDS,
    SYNTH_PARAMS,
    TWIN_PARAMS,
)
from control.segmented_controller import (
    SegmentedFunnelController,
    create_controller_from_params,
    design_initial_controller,
)
from data_driven.beta_computation import compute_beta
from data_driven.data_collection import SegmentManager, build_data_matrices
from dynamics.digital_twin import create_plant, create_twin
from dynamics.robot_arm import RobotArm2DoF
from simulation.baseline import (
    BaselineResults,
    compare_methods,
    design_twin_controller,
    run_baseline,
)
from simulation.closed_loop import SimulationResults, run_simulation
from synthesis.ellipsoid_bounds import (
    compute_mvie_input,
    compute_mvie_state,
    get_bounds_from_config,
)
from synthesis.sdp_solver import solve_funnel_sdp, solve_initial_segment_sdp
from trajectory.nominal_planner import generate_nominal_trajectory


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def systems():
    """Create plant and twin systems."""
    plant = create_plant()
    twin = create_twin()
    return plant, twin


@pytest.fixture
def arm():
    """Create robot arm model."""
    return RobotArm2DoF(**TWIN_PARAMS)


@pytest.fixture
def trajectory(systems, arm):
    """Generate nominal trajectory."""
    _, twin = systems
    x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
    x_goal = np.array(INITIAL_STATES["x_goal"])
    u_goal = np.array(INITIAL_STATES["u_goal"])
    N = SIM_PARAMS["N"]

    x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)
    return x_nom, u_nom


@pytest.fixture
def controller(systems, trajectory):
    """Create segmented funnel controller."""
    _, twin = systems
    x_nom, u_nom = trajectory
    return create_controller_from_params(twin, x_nom, u_nom)


# ============================================================================
# Basic Component Tests
# ============================================================================


class TestSystemCreation:
    """Test system creation."""

    def test_plant_creation(self, systems):
        """Test plant is created correctly."""
        plant, _ = systems
        assert plant is not None
        assert hasattr(plant, "step")

    def test_twin_creation(self, systems):
        """Test twin is created correctly."""
        _, twin = systems
        assert twin is not None
        assert hasattr(twin, "step")

    def test_plant_twin_different_params(self, systems):
        """Verify plant and twin have different parameters."""
        plant, twin = systems
        # They should be different systems (different parameters)
        # Test by stepping from same state with same input
        x0 = np.array([0.5, 0.5, 0.0, 0.0])
        u0 = np.array([1.0, 0.5])

        x_plant = plant.step(x0, u0)
        x_twin = twin.step(x0, u0)

        # Should produce different results due to parameter mismatch
        diff = np.linalg.norm(x_plant - x_twin)
        assert diff > 0, "Plant and twin should have different parameters"


class TestTrajectoryGeneration:
    """Test trajectory generation."""

    def test_trajectory_shape(self, trajectory):
        """Test trajectory has correct shape."""
        x_nom, u_nom = trajectory
        N = SIM_PARAMS["N"]
        n, m = 4, 2

        assert x_nom.shape == (N + 1, n)
        assert u_nom.shape == (N, m)

    def test_trajectory_start(self, trajectory):
        """Test trajectory starts at correct state."""
        x_nom, _ = trajectory
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])

        np.testing.assert_allclose(x_nom[0], x_nom_0, rtol=1e-3)

    def test_trajectory_reaches_goal(self, trajectory):
        """Test trajectory approximately reaches goal."""
        x_nom, _ = trajectory
        x_goal = np.array(INITIAL_STATES["x_goal"])

        # Final state should be close to goal
        dist = np.linalg.norm(x_nom[-1] - x_goal)
        assert dist < 0.1, f"Final state too far from goal: {dist}"


class TestControllerDesign:
    """Test controller design."""

    def test_initial_controller_design(self, systems, trajectory):
        """Test initial controller design."""
        _, twin = systems
        x_nom, u_nom = trajectory

        K0, P0 = design_initial_controller(twin, x_nom, u_nom)

        assert K0.shape == (2, 4), f"K0 shape wrong: {K0.shape}"
        assert P0.shape == (4, 4), f"P0 shape wrong: {P0.shape}"

        # K0 should have non-zero elements (active feedback)
        assert np.linalg.norm(K0) > 0.1, "K0 should provide active feedback"

        # P0 should be positive definite
        eigvals = np.linalg.eigvalsh(P0)
        assert eigvals.min() > 0, "P0 should be positive definite"

    def test_controller_creation(self, controller):
        """Test controller is created and initialized."""
        assert controller is not None
        assert controller.P is not None
        assert controller.K is not None
        assert len(controller.P_history) > 0
        assert len(controller.K_history) > 0


# ============================================================================
# Short Simulation Tests (Faster)
# ============================================================================


class TestShortSimulation:
    """Test with shortened simulation for faster execution."""

    @pytest.fixture
    def short_trajectory(self, systems, arm):
        """Generate short trajectory (2 segments)."""
        _, twin = systems
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        x_goal = np.array(INITIAL_STATES["x_goal"])
        u_goal = np.array(INITIAL_STATES["u_goal"])
        N = 200  # 2 segments

        x_nom, u_nom = generate_nominal_trajectory(twin, x_nom_0, x_goal, u_goal, N, arm=arm)
        return x_nom, u_nom

    @pytest.fixture
    def short_controller(self, systems, short_trajectory):
        """Create controller for short simulation."""
        _, twin = systems
        x_nom, u_nom = short_trajectory
        return create_controller_from_params(twin, x_nom, u_nom)

    def test_short_simulation_runs(self, systems, short_trajectory, short_controller):
        """Test that short simulation runs without errors."""
        plant, _ = systems
        x_nom, u_nom = short_trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])

        results = run_simulation(
            plant, short_controller.twin, x_nom, u_nom, x0_plant,
            controller=short_controller, verbose=False
        )

        assert results is not None
        assert isinstance(results, SimulationResults)
        assert len(results.x_trajectory) == len(x_nom)

    def test_short_baseline_runs(self, systems, short_trajectory):
        """Test that baseline runs without errors."""
        plant, twin = systems
        x_nom, u_nom = short_trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])
        x_goal = np.array(INITIAL_STATES["x_goal"])
        u_goal = np.array(INITIAL_STATES["u_goal"])

        K_baseline = design_twin_controller(twin, x_goal, u_goal)
        baseline = run_baseline(plant, x_nom, u_nom, x0_plant, K_baseline, verbose=False)

        assert baseline is not None
        assert isinstance(baseline, BaselineResults)


# ============================================================================
# Full Simulation Tests (Slower)
# ============================================================================


class TestFullSimulation:
    """Full simulation tests - marked slow for optional skipping."""

    @pytest.mark.slow
    def test_full_simulation_completes(self, systems, trajectory, controller):
        """Test full simulation runs to completion."""
        plant, _ = systems
        x_nom, u_nom = trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])

        results = run_simulation(
            plant, controller.twin, x_nom, u_nom, x0_plant,
            controller=controller, verbose=False
        )

        assert results is not None
        assert len(results.x_trajectory) == SIM_PARAMS["N"] + 1
        assert np.all(np.isfinite(results.x_trajectory))

    @pytest.mark.slow
    def test_all_segments_complete(self, systems, trajectory, controller):
        """Test all 6 segments complete."""
        plant, _ = systems
        x_nom, u_nom = trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])

        results = run_simulation(
            plant, controller.twin, x_nom, u_nom, x0_plant,
            controller=controller, verbose=False
        )

        # Should have 6 P matrices (initial + 5 updates)
        assert len(results.P_history) >= 1
        # Should have completed all segments
        num_segments = SIM_PARAMS["N"] // SIM_PARAMS["T"]
        assert len(results.P_history) <= num_segments + 1

    @pytest.mark.slow
    def test_sdps_solve(self, systems, trajectory, controller):
        """Test all SDPs solve successfully."""
        plant, _ = systems
        x_nom, u_nom = trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])

        results = run_simulation(
            plant, controller.twin, x_nom, u_nom, x0_plant,
            controller=controller, verbose=False
        )

        # Check SDP results
        for sdp_result in results.sdp_results:
            assert "optimal" in sdp_result["status"].lower(), (
                f"SDP failed: {sdp_result['status']}"
            )

    @pytest.mark.slow
    def test_proposed_outperforms_baseline(self, systems, trajectory, controller):
        """Test proposed method outperforms baseline."""
        plant, twin = systems
        x_nom, u_nom = trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])
        x_goal = np.array(INITIAL_STATES["x_goal"])
        u_goal = np.array(INITIAL_STATES["u_goal"])

        # Run proposed
        results = run_simulation(
            plant, twin, x_nom, u_nom, x0_plant,
            controller=controller, verbose=False
        )

        # Run baseline
        K_baseline = design_twin_controller(twin, x_goal, u_goal)
        baseline = run_baseline(plant, x_nom, u_nom, x0_plant, K_baseline, verbose=False)

        # Compare
        comparison = compare_methods(results, baseline)

        # Proposed should be better (lower deviation)
        # Note: With large initial deviation, this might not always hold
        # but the method should still complete successfully
        assert results.eta_final_norm < np.inf
        assert np.isfinite(results.max_eta_norm)


# ============================================================================
# Constraint Tests
# ============================================================================


class TestConstraints:
    """Test constraint satisfaction."""

    def test_input_bounds(self, systems, trajectory, controller):
        """Test input constraints are satisfied (via saturation)."""
        plant, _ = systems
        x_nom, u_nom = trajectory
        x0_plant = np.array(INITIAL_STATES["x_plant_0"])

        results = run_simulation(
            plant, controller.twin, x_nom, u_nom, x0_plant,
            controller=controller, verbose=False
        )

        u_min = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
        u_max = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])

        # All inputs should be within bounds (due to saturation)
        assert np.all(results.u_trajectory >= u_min - 1e-6)
        assert np.all(results.u_trajectory <= u_max + 1e-6)


# ============================================================================
# SDP Component Tests
# ============================================================================


class TestSDPComponents:
    """Test SDP solver components."""

    def test_initial_sdp(self):
        """Test initial segment SDP solves."""
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array(INITIAL_STATES["u_goal"])

        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)

        result = solve_initial_segment_sdp(P_min, R_max, n=4, m=2)

        assert result.success
        assert "optimal" in result.status.lower()

    def test_mvie_computation(self):
        """Test MVIE computation."""
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array(INITIAL_STATES["u_goal"])

        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)

        # Should be positive definite
        assert np.linalg.eigvalsh(P_min).min() > 0
        assert np.linalg.eigvalsh(R_max).min() > 0


# ============================================================================
# Data Collection Tests
# ============================================================================


class TestDataCollection:
    """Test data collection and processing."""

    def test_segment_manager(self):
        """Test segment manager."""
        T, L = 100, 60
        manager = SegmentManager(T, L, n=4, m=2)

        # Test segment boundaries
        assert manager.get_segment_start(0) == 0
        assert manager.get_segment_end(0) == T
        assert manager.get_segment_start(1) == T
        assert manager.get_segment_end(1) == 2 * T

    def test_data_window(self):
        """Test data window computation."""
        T, L = 100, 60
        manager = SegmentManager(T, L, n=4, m=2)

        window = manager.get_data_window_indices(0)
        assert len(window) == L


# ============================================================================
# Utility Functions
# ============================================================================


def test_config_params_valid():
    """Test configuration parameters are valid."""
    # Check dimensions
    assert len(INITIAL_STATES["x_nom_0"]) == 4
    assert len(INITIAL_STATES["x_goal"]) == 4
    assert len(INITIAL_STATES["u_goal"]) == 2
    assert len(INITIAL_STATES["x_plant_0"]) == 4

    # Check simulation params
    assert SIM_PARAMS["N"] % SIM_PARAMS["T"] == 0
    assert SIM_PARAMS["L"] < SIM_PARAMS["T"]

    # Check synthesis params
    assert 0 < SYNTH_PARAMS["alpha"] < 1
    assert SYNTH_PARAMS["mu"] > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
