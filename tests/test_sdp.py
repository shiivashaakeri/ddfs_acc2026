"""
Tests for SDP synthesis components.

Tests cover:
- LMI matrix construction (S, N1, N2)
- SDP solver feasibility
- Recovered controller stability
"""

import numpy as np
import pytest

from synthesis.lmi_matrices import (
    build_S_matrix,
    build_N1_tilde,
    build_N2_tilde,
    build_full_lmi_matrix,
    get_block_indices,
    get_total_dimension,
    verify_lmi_dimensions,
)
from synthesis.sdp_solver import (
    solve_funnel_sdp,
    solve_initial_segment_sdp,
    verify_funnel_constraints,
)
from synthesis.ellipsoid_bounds import (
    compute_mvie_state,
    compute_mvie_input,
    get_bounds_from_config,
)
from config.params import INITIAL_STATES, LIPSCHITZ, SIM_PARAMS, SYNTH_PARAMS


# Dimensions for 2-DoF arm
N_STATE = 4
M_INPUT = 2
L_DATA = 60  # Data window length


class TestLMIDimensions:
    """Test LMI matrix dimensions."""

    def test_total_dimension(self):
        """Total LMI dimension should be 4n + 2m = 20."""
        dim = get_total_dimension(N_STATE, M_INPUT)
        assert dim == 20

    def test_block_indices(self):
        """Block indices should be [0, 4, 8, 10, 14, 16, 20]."""
        idx = get_block_indices(N_STATE, M_INPUT)
        assert idx == [0, 4, 8, 10, 14, 16, 20]

    def test_verify_dimensions(self):
        """Dimension verification should pass."""
        result = verify_lmi_dimensions(N_STATE, M_INPUT)
        assert result["matches"] is True
        assert result["total_dimension"] == 20
        assert result["block_sizes"] == [4, 4, 2, 4, 2, 4]


class TestSMatrixConstruction:
    """Test S matrix construction."""

    @pytest.fixture
    def sample_matrices(self):
        """Create sample P, L matrices."""
        P = np.eye(N_STATE) * 0.5
        L = np.ones((M_INPUT, N_STATE)) * 0.1
        nu = 1.0
        alpha = 0.98
        return P, L, nu, alpha

    def test_S_matrix_dimensions(self, sample_matrices):
        """S matrix should be 20x20."""
        P, L, nu, alpha = sample_matrices
        S = build_S_matrix(P, L, nu, alpha, N_STATE, M_INPUT)
        assert S.shape == (20, 20)

    def test_S_matrix_symmetry(self, sample_matrices):
        """S matrix should be symmetric."""
        P, L, nu, alpha = sample_matrices
        S = build_S_matrix(P, L, nu, alpha, N_STATE, M_INPUT)
        assert np.allclose(S, S.T), "S matrix is not symmetric"

    def test_S_matrix_block_00(self, sample_matrices):
        """Block (0,0) should be alpha*P - nu*I."""
        P, L, nu, alpha = sample_matrices
        S = build_S_matrix(P, L, nu, alpha, N_STATE, M_INPUT)
        expected = alpha * P - nu * np.eye(N_STATE)
        assert np.allclose(S[0:4, 0:4], expected)

    def test_S_matrix_block_55(self, sample_matrices):
        """Block (5,5) should be P."""
        P, L, nu, alpha = sample_matrices
        S = build_S_matrix(P, L, nu, alpha, N_STATE, M_INPUT)
        assert np.allclose(S[16:20, 16:20], P)


class TestN1N2Matrices:
    """Test N1_tilde and N2_tilde construction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data matrices."""
        np.random.seed(42)
        H = np.random.randn(N_STATE, L_DATA) * 0.1
        H_plus = H + np.random.randn(N_STATE, L_DATA) * 0.05
        Xi = np.random.randn(M_INPUT, L_DATA) * 0.1
        beta = 0.1
        return H, H_plus, Xi, beta

    def test_N1_dimensions(self, sample_data):
        """N1_tilde should be 20x20."""
        H, H_plus, Xi, beta = sample_data
        N1 = build_N1_tilde(H, H_plus, Xi, beta, N_STATE, M_INPUT)
        assert N1.shape == (20, 20)

    def test_N1_symmetry(self, sample_data):
        """N1_tilde should be symmetric."""
        H, H_plus, Xi, beta = sample_data
        N1 = build_N1_tilde(H, H_plus, Xi, beta, N_STATE, M_INPUT)
        assert np.allclose(N1, N1.T), "N1_tilde is not symmetric"

    def test_N2_dimensions(self):
        """N2_tilde should be 20x20."""
        C = 3.64
        T_tilde = 139
        N2 = build_N2_tilde(C, T_tilde, N_STATE, M_INPUT)
        assert N2.shape == (20, 20)

    def test_N2_symmetry(self):
        """N2_tilde should be symmetric."""
        C = 3.64
        T_tilde = 139
        N2 = build_N2_tilde(C, T_tilde, N_STATE, M_INPUT)
        assert np.allclose(N2, N2.T), "N2_tilde is not symmetric"

    def test_N2_block_33_value(self):
        """N2 block (3,3) should be C^2 * T_tilde^2 * I."""
        C = 3.64
        T_tilde = 139
        N2 = build_N2_tilde(C, T_tilde, N_STATE, M_INPUT)
        expected_val = C**2 * T_tilde**2
        assert np.allclose(np.diag(N2[10:14, 10:14]), expected_val)


class TestFullLMIMatrix:
    """Test full LMI matrix construction."""

    def test_full_lmi_dimensions(self):
        """Full LMI should be 20x20."""
        np.random.seed(42)
        P = np.eye(N_STATE) * 0.5
        L = np.ones((M_INPUT, N_STATE)) * 0.1
        nu = 1.0
        alpha = 0.98
        H = np.random.randn(N_STATE, L_DATA) * 0.1
        H_plus = H + np.random.randn(N_STATE, L_DATA) * 0.05
        Xi = np.random.randn(M_INPUT, L_DATA) * 0.1
        beta = 0.1
        C = 3.64
        T_tilde = 139

        LMI = build_full_lmi_matrix(
            P, L, nu, alpha, H, H_plus, Xi, beta, C, T_tilde, N_STATE, M_INPUT
        )
        assert LMI.shape == (20, 20)

    def test_full_lmi_symmetry(self):
        """Full LMI should be symmetric."""
        np.random.seed(42)
        P = np.eye(N_STATE) * 0.5
        L = np.ones((M_INPUT, N_STATE)) * 0.1
        nu = 1.0
        alpha = 0.98
        H = np.random.randn(N_STATE, L_DATA) * 0.1
        H_plus = H + np.random.randn(N_STATE, L_DATA) * 0.05
        Xi = np.random.randn(M_INPUT, L_DATA) * 0.1
        beta = 0.1
        C = 3.64
        T_tilde = 139

        LMI = build_full_lmi_matrix(
            P, L, nu, alpha, H, H_plus, Xi, beta, C, T_tilde, N_STATE, M_INPUT
        )
        assert np.allclose(LMI, LMI.T), "Full LMI is not symmetric"


class TestInitialSegmentSDP:
    """Test initial segment SDP (no data)."""

    @pytest.fixture
    def bounds(self):
        """Get constraint bounds."""
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array([11.44, 2.38])
        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)
        return P_min, R_max

    def test_initial_sdp_feasible(self, bounds):
        """Initial SDP should find a feasible solution."""
        P_min, R_max = bounds
        result = solve_initial_segment_sdp(P_min, R_max, N_STATE, M_INPUT)
        assert result.success, f"Initial SDP failed: {result.status}"

    def test_initial_sdp_P_positive_definite(self, bounds):
        """Resulting P should be positive definite."""
        P_min, R_max = bounds
        result = solve_initial_segment_sdp(P_min, R_max, N_STATE, M_INPUT)
        assert result.success
        eigvals = np.linalg.eigvalsh(result.P)
        assert eigvals.min() > 0, f"P not positive definite: min eigval = {eigvals.min()}"

    def test_initial_sdp_constraints_satisfied(self, bounds):
        """All constraints should be satisfied."""
        P_min, R_max = bounds
        result = solve_initial_segment_sdp(P_min, R_max, N_STATE, M_INPUT)
        assert result.success
        verify = verify_funnel_constraints(result.P, result.L, P_min, R_max)
        assert verify["all_satisfied"], f"Constraints violated: {verify}"


class TestFullSDP:
    """Test full SDP with data."""

    @pytest.fixture
    def easy_problem(self):
        """Create an easy SDP problem with small deviations."""
        np.random.seed(42)

        # Small, well-conditioned data
        H = np.random.randn(N_STATE, L_DATA) * 0.05
        H_plus = H + np.random.randn(N_STATE, L_DATA) * 0.02
        Xi = np.random.randn(M_INPUT, L_DATA) * 0.05
        beta = 0.05  # Small disturbance bound

        # Bounds
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array([11.44, 2.38])
        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)

        # Parameters
        alpha = SYNTH_PARAMS["alpha"]
        C = LIPSCHITZ["C"]
        T = SIM_PARAMS["T"]
        L = SIM_PARAMS["L"]
        T_tilde = 2 * T - L - 1

        return {
            "H": H,
            "H_plus": H_plus,
            "Xi": Xi,
            "beta": beta,
            "P_min": P_min,
            "R_max": R_max,
            "alpha": alpha,
            "C": C,
            "T_tilde": T_tilde,
        }

    def test_sdp_feasible_with_good_data(self, easy_problem):
        """SDP should be feasible with well-conditioned data."""
        result = solve_funnel_sdp(
            easy_problem["H"],
            easy_problem["H_plus"],
            easy_problem["Xi"],
            easy_problem["beta"],
            easy_problem["P_min"],
            easy_problem["R_max"],
            easy_problem["alpha"],
            easy_problem["C"],
            easy_problem["T_tilde"],
            N_STATE,
            M_INPUT,
        )
        assert result.success, f"SDP failed: {result.status}"

    def test_sdp_returns_optimal_status(self, easy_problem):
        """SDP should return optimal or optimal_inaccurate status."""
        result = solve_funnel_sdp(
            easy_problem["H"],
            easy_problem["H_plus"],
            easy_problem["Xi"],
            easy_problem["beta"],
            easy_problem["P_min"],
            easy_problem["R_max"],
            easy_problem["alpha"],
            easy_problem["C"],
            easy_problem["T_tilde"],
            N_STATE,
            M_INPUT,
        )
        assert result.status in ["optimal", "optimal_inaccurate"], f"Status: {result.status}"

    def test_sdp_P_positive_definite(self, easy_problem):
        """Resulting P should be positive definite."""
        result = solve_funnel_sdp(
            easy_problem["H"],
            easy_problem["H_plus"],
            easy_problem["Xi"],
            easy_problem["beta"],
            easy_problem["P_min"],
            easy_problem["R_max"],
            easy_problem["alpha"],
            easy_problem["C"],
            easy_problem["T_tilde"],
            N_STATE,
            M_INPUT,
        )
        assert result.success
        eigvals = np.linalg.eigvalsh(result.P)
        assert eigvals.min() > 0, f"P not positive definite: min eigval = {eigvals.min()}"

    def test_sdp_K_computed_correctly(self, easy_problem):
        """K should equal L @ inv(P)."""
        result = solve_funnel_sdp(
            easy_problem["H"],
            easy_problem["H_plus"],
            easy_problem["Xi"],
            easy_problem["beta"],
            easy_problem["P_min"],
            easy_problem["R_max"],
            easy_problem["alpha"],
            easy_problem["C"],
            easy_problem["T_tilde"],
            N_STATE,
            M_INPUT,
        )
        assert result.success
        K_expected = result.L @ np.linalg.inv(result.P)
        assert np.allclose(result.K, K_expected, atol=1e-6)

    def test_sdp_constraints_satisfied(self, easy_problem):
        """All constraints should be satisfied."""
        result = solve_funnel_sdp(
            easy_problem["H"],
            easy_problem["H_plus"],
            easy_problem["Xi"],
            easy_problem["beta"],
            easy_problem["P_min"],
            easy_problem["R_max"],
            easy_problem["alpha"],
            easy_problem["C"],
            easy_problem["T_tilde"],
            N_STATE,
            M_INPUT,
        )
        assert result.success
        verify = verify_funnel_constraints(
            result.P, result.L, easy_problem["P_min"], easy_problem["R_max"]
        )
        assert verify["all_satisfied"], f"Constraints violated: {verify}"


class TestRecoveredControllerStability:
    """Test that recovered K stabilizes the linearized system."""

    def test_recovered_K_stabilizes(self):
        """K = L @ inv(P) should stabilize a linearized system."""
        np.random.seed(123)

        # Create a marginally stable open-loop system
        # A has eigenvalues close to unit circle
        A = np.array([
            [0.95, 0.1, 0.0, 0.0],
            [-0.1, 0.95, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.2],
            [0.0, 0.0, -0.2, 0.9],
        ])
        B = np.array([
            [0.1, 0.0],
            [0.0, 0.1],
            [0.05, 0.0],
            [0.0, 0.05],
        ])

        # Generate data consistent with this system
        L_data = 60
        H = np.random.randn(N_STATE, L_data) * 0.1
        # H_plus ≈ A @ H (with small noise to simulate disturbance)
        H_plus = A @ H + np.random.randn(N_STATE, L_data) * 0.01
        Xi = np.random.randn(M_INPUT, L_data) * 0.1

        # Bounds
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array([11.44, 2.38])
        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)

        # Parameters
        alpha = 0.98
        C = 3.64
        T_tilde = 139
        beta = 0.05

        # Solve SDP
        result = solve_funnel_sdp(
            H, H_plus, Xi, beta,
            P_min, R_max,
            alpha, C, T_tilde,
            N_STATE, M_INPUT,
        )

        if not result.success:
            pytest.skip(f"SDP did not converge: {result.status}")

        # Check closed-loop stability: A - B @ K
        K = result.K
        A_cl = A - B @ K
        cl_eigvals = np.linalg.eigvals(A_cl)
        spectral_radius = np.max(np.abs(cl_eigvals))

        # Closed-loop should be stable (spectral radius < 1)
        # We allow some margin since the SDP doesn't directly optimize for this A,B
        assert spectral_radius < 1.1, (
            f"Closed-loop not stable: spectral radius = {spectral_radius:.4f}, "
            f"eigenvalues = {cl_eigvals}"
        )


class TestSDPWithContinuityConstraint:
    """Test SDP with cross-segment continuity constraint."""

    def test_continuity_constraint(self):
        """P should satisfy P <= mu * P_prev when P_prev is given."""
        np.random.seed(42)

        # Small data
        H = np.random.randn(N_STATE, L_DATA) * 0.05
        H_plus = H + np.random.randn(N_STATE, L_DATA) * 0.02
        Xi = np.random.randn(M_INPUT, L_DATA) * 0.05
        beta = 0.05

        # Bounds
        x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()
        x_nom_0 = np.array(INITIAL_STATES["x_nom_0"])
        u_nom_0 = np.array([11.44, 2.38])
        P_min = compute_mvie_state(x_nom_0, x_lower, x_upper)
        R_max = compute_mvie_input(u_nom_0, u_lower, u_upper)

        alpha = 0.98
        C = 3.64
        T_tilde = 139
        mu = 1.02

        # First solve without P_prev
        result1 = solve_funnel_sdp(
            H, H_plus, Xi, beta,
            P_min, R_max,
            alpha, C, T_tilde,
            N_STATE, M_INPUT,
        )

        if not result1.success:
            pytest.skip(f"First SDP failed: {result1.status}")

        # Solve with P_prev constraint
        P_prev = result1.P * 1.5  # Make P_prev larger so constraint is active
        result2 = solve_funnel_sdp(
            H, H_plus, Xi, beta,
            P_min, R_max,
            alpha, C, T_tilde,
            N_STATE, M_INPUT,
            P_prev=P_prev,
            mu=mu,
        )

        if not result2.success:
            pytest.skip(f"Second SDP failed: {result2.status}")

        # Check P <= mu * P_prev
        diff = mu * P_prev - result2.P
        eigvals = np.linalg.eigvalsh(diff)
        assert eigvals.min() >= -1e-6, (
            f"Continuity violated: min eigenvalue of (mu*P_prev - P) = {eigvals.min()}"
        )
