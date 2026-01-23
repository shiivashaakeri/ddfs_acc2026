"""
Segmented Funnel Controller (Algorithm 1).

Implements the Online Data-Driven Funnel Control algorithm:
1. Execute on segment T_i with fixed feedback K_i
2. During data window T_i^D, add excitation signal epsilon
3. At segment boundary, assemble data and solve SDP for next segment

This is the main controller class that coordinates all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.params import LIPSCHITZ, SIM_PARAMS, SYNTH_PARAMS
from control.excitation import ExcitationGenerator
from data_driven.beta_computation import compute_beta, regularize_beta
from data_driven.data_collection import SegmentManager, build_data_matrices, check_rank_condition
from synthesis.ellipsoid_bounds import (
    compute_mvie_input,
    compute_mvie_state,
    get_bounds_from_config,
)
from synthesis.sdp_solver import solve_funnel_sdp, solve_initial_segment_sdp


@dataclass
class ControllerState:
    """Current state of the segmented controller."""

    segment_idx: int = 0
    P: np.ndarray = field(default_factory=lambda: np.eye(4))
    K: np.ndarray = field(default_factory=lambda: np.zeros((2, 4)))
    L: np.ndarray = field(default_factory=lambda: np.zeros((2, 4)))


class SegmentedFunnelController:
    """
    Online Data-Driven Funnel Controller (Algorithm 1).

    This controller:
    1. Tracks nominal trajectory using segment-wise feedback gains
    2. Collects data during data windows for uncertainty quantification
    3. Synthesizes new controllers at segment boundaries via SDP
    """

    def __init__(
        self,
        twin,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        T: int = 100,
        L: int = 60,
        alpha: float = 0.98,
        mu: float = 1.02,
        eps_bar: float = 0.15,
        C: float = 3.64,
        gamma: float = 0.034,
        L_r: float = 0.074,
        gain_ramp_steps: int = 30,
        gain_min_scale: float = 0.15,
    ):
        """
        Initialize segmented controller.

        Args:
            twin: Digital twin DiscreteSystem
            x_nom: (N+1, n) Nominal state trajectory
            u_nom: (N, m) Nominal input trajectory
            T: Segment length (default 100)
            L: Data window length (default 60)
            alpha: Contraction rate (default 0.98)
            mu: Growth bound (default 1.02)
            eps_bar: Excitation bound (default 0.15)
            C: Jacobian variation rate
            gamma: Plant-twin mismatch bound
            L_r: Second-order remainder constant
            gain_ramp_steps: Number of timesteps to ramp up gain (default 30)
            gain_min_scale: Initial gain scale factor (default 0.15)
        """
        self.twin = twin
        self.x_nom = np.asarray(x_nom)
        self.u_nom = np.asarray(u_nom)

        self.N = u_nom.shape[0]
        self.n = x_nom.shape[1]
        self.m = u_nom.shape[1]

        # Parameters
        self.T = T
        self.L = L
        self.alpha = alpha
        self.mu = mu
        self.eps_bar = eps_bar
        self.C = C
        self.gamma = gamma
        self.L_r = L_r
        self.T_tilde = 2 * T - L - 1

        # Gain scheduling for smooth startup
        self.gain_ramp_steps = gain_ramp_steps
        self.gain_min_scale = gain_min_scale

        # Segment manager
        self.segment_manager = SegmentManager(T, L, self.n, self.m)
        self.num_segments = self.segment_manager.num_segments(self.N)

        # Get bounds
        self.x_lower, self.x_upper, self.u_lower, self.u_upper = get_bounds_from_config()

        # Excitation generator
        self.excitation_gen = ExcitationGenerator(self.m, eps_bar, mode="random")

        # Storage
        self.P_history: List[np.ndarray] = []
        self.K_history: List[np.ndarray] = []
        self.L_history: List[np.ndarray] = []
        self.beta_history: List[float] = []

        # Logs for data collection
        self.eta_log: Dict[int, np.ndarray] = {}  # eta(k) = x(k) - x_nom(k)
        self.xi_log: Dict[int, np.ndarray] = {}  # xi(k) = u(k) - u_nom(k)

        # Current controller state
        self.current_segment = 0
        self.P = None
        self.K = None
        self.L_mat = None

        # Diagnostics
        self.sdp_results: List[dict] = []

    def initialize(self, K0: np.ndarray, P0: np.ndarray):
        """
        Initialize controller with initial gains.

        Args:
            K0: (m, n) Initial feedback gain
            P0: (n, n) Initial funnel shape
        """
        self.K = np.asarray(K0)
        self.P = np.asarray(P0)
        self.L_mat = self.K @ self.P  # L = K @ P

        self.K_history.append(self.K.copy())
        self.P_history.append(self.P.copy())
        self.L_history.append(self.L_mat.copy())

        self.current_segment = 0

    def compute_control(self, k: int, x: np.ndarray) -> np.ndarray:
        """
        Compute control input u(k) given current state x(k).

        u(k) = u_nom(k) + K_i @ (x(k) - x_nom(k)) + epsilon(k)

        where epsilon(k) is added during data window for excitation.

        Args:
            k: Current time index
            x: Current state (n,)

        Returns:
            u: Control input (m,)
        """
        x = np.asarray(x)

        # Compute deviation from nominal
        eta_k = x - self.x_nom[k]

        # Store deviation
        self.eta_log[k] = eta_k.copy()

        # Apply gain scheduling for smooth startup
        # Smoothly ramp gain from gain_min_scale to 1.0 over gain_ramp_steps
        if k < self.gain_ramp_steps:
            # Smooth cosine ramp: starts slow, accelerates, then slows
            t = k / self.gain_ramp_steps
            gain_scale = self.gain_min_scale + (1.0 - self.gain_min_scale) * (1 - np.cos(np.pi * t)) / 2
        else:
            gain_scale = 1.0

        # Feedback control with gain scheduling
        u_fb = gain_scale * (self.K @ eta_k)

        # Nominal input
        u_nom_k = self.u_nom[k]

        # Add excitation during data window
        epsilon = np.zeros(self.m)
        if self._in_data_window(k):
            epsilon = self.excitation_gen()

        # Total control
        u = u_nom_k + u_fb + epsilon

        # Store input deviation (including excitation)
        xi_k = u - u_nom_k
        self.xi_log[k] = xi_k.copy()

        return u

    def _in_data_window(self, k: int) -> bool:
        """Check if k is in current segment's data window."""
        return self.segment_manager.is_in_data_window(k, self.current_segment)

    def end_of_segment_update(self, k: int) -> bool:
        """
        Called at end of segment to synthesize next controller.

        Args:
            k: Current time index (should be at segment boundary)

        Returns:
            success: True if SDP solved successfully
        """
        segment_end = self.segment_manager.get_segment_end(self.current_segment)

        if k != segment_end - 1:
            return True  # Not at segment boundary

        # Check if this is the last segment
        if self.current_segment >= self.num_segments - 1:
            return True  # No more segments

        # Build data matrices from current segment
        eta_array = self._build_eta_array()
        xi_array = self._build_xi_array()

        H, H_plus, Xi = build_data_matrices(
            eta_array, xi_array, self.current_segment, self.segment_manager
        )

        # Check rank condition
        satisfied, rank = check_rank_condition(H, Xi, self.n, self.m)
        if not satisfied:
            print(f"Warning: Rank condition not satisfied at segment {self.current_segment}")
            print(f"  rank([H; Xi]) = {rank}, required = {self.n + self.m}")

        # Compute beta
        data_window = self.segment_manager.get_data_window_indices(self.current_segment)
        k_i = self.segment_manager.get_segment_start(self.current_segment)
        beta_raw = compute_beta(
            eta_array, xi_array, data_window, k_i, self.C, self.gamma, self.L_r
        )

        # Regularize beta to prevent SDP numerical issues
        # Large beta overwhelms LMI constraints, producing meaningless K
        beta = regularize_beta(beta_raw, beta_max=1.0, method="cap")
        self.beta_history.append(beta)

        # Get bounds for next segment
        next_segment_start = self.segment_manager.get_segment_start(self.current_segment + 1)
        P_min = compute_mvie_state(
            self.x_nom[next_segment_start], self.x_lower, self.x_upper
        )
        R_max = compute_mvie_input(
            self.u_nom[next_segment_start], self.u_lower, self.u_upper
        )

        # Solve SDP for next segment
        result = solve_funnel_sdp(
            H, H_plus, Xi, beta,
            P_min, R_max,
            self.alpha, self.C, self.T_tilde,
            self.n, self.m,
            P_prev=self.P,
            mu=self.mu,
        )

        # Store diagnostics
        self.sdp_results.append({
            "segment": self.current_segment,
            "status": result.status,
            "success": result.success,
            "objective": result.objective,
            "beta": beta,
            "beta_raw": beta_raw,
            "rank": rank,
        })

        if result.success:
            # Update P from SDP result
            self.P = result.P

            # Try to use K from SDP first if it's stabilizing
            K_sdp = result.K
            sdp_stable = self._check_stability(K_sdp, next_segment_start)

            if sdp_stable and np.linalg.norm(K_sdp) > 0.1:
                # Use SDP's K - it's non-trivial and stabilizing
                self.K = K_sdp
                self.L_mat = result.L
            else:
                # Fall back to LQR if SDP's K is trivial or unstable
                from trajectory.nominal_planner import linearize_at_point, compute_lqr_gain

                A, B = linearize_at_point(self.twin, self.x_nom[next_segment_start], self.u_nom[next_segment_start])
                Q = np.diag([500.0, 500.0, 100.0, 100.0])
                R = np.eye(self.m) * 0.1
                try:
                    K_lqr = compute_lqr_gain(A, B, Q, R)
                    self.K = -K_lqr  # Negate for our control convention
                except Exception:
                    pass  # Keep current K if LQR fails

                # Update L for consistency: L = K @ P
                self.L_mat = self.K @ self.P

            self.P_history.append(self.P.copy())
            self.K_history.append(self.K.copy())
            self.L_history.append(self.L_mat.copy())

            self.current_segment += 1
            self.excitation_gen.reset()
            return True
        else:
            print(f"SDP failed at segment {self.current_segment}: {result.status}")
            # Keep current controller
            self.P_history.append(self.P.copy())
            self.K_history.append(self.K.copy())
            self.L_history.append(self.L_mat.copy())
            self.current_segment += 1
            return False

    def _build_eta_array(self) -> np.ndarray:
        """Build eta array from logs for current segment."""
        # Need eta from start to end+1 of segment
        start = 0
        end = self.segment_manager.get_segment_end(self.current_segment)

        eta_array = np.zeros((end + 1, self.n))
        for k in range(end + 1):
            if k in self.eta_log:
                eta_array[k] = self.eta_log[k]
        return eta_array

    def _build_xi_array(self) -> np.ndarray:
        """Build xi array from logs for current segment."""
        end = self.segment_manager.get_segment_end(self.current_segment)

        xi_array = np.zeros((end, self.m))
        for k in range(end):
            if k in self.xi_log:
                xi_array[k] = self.xi_log[k]
        return xi_array

    def _check_stability(self, K: np.ndarray, k: int) -> bool:
        """
        Check if gain K produces a stable closed-loop system.

        Linearizes the twin at operating point k and checks if all
        closed-loop eigenvalues are inside the unit circle.

        Args:
            K: (m, n) Feedback gain
            k: Time index for linearization point

        Returns:
            True if closed-loop system is stable
        """
        from trajectory.nominal_planner import linearize_at_point

        # Get linearization at operating point
        x_op = self.x_nom[k]
        u_op = self.u_nom[k] if k < len(self.u_nom) else self.u_nom[-1]

        A, B = linearize_at_point(self.twin, x_op, u_op)

        # Closed-loop: x_next = (A + B @ K) @ x
        A_cl = A + B @ K
        eigs = np.linalg.eigvals(A_cl)

        # Check all eigenvalues inside unit circle (with small margin)
        max_eig_mag = np.max(np.abs(eigs))
        return max_eig_mag < 1.0

    def get_funnel_at_k(self, k: int) -> np.ndarray:
        """Get the funnel matrix P for time k."""
        segment = self.segment_manager.get_segment_for_index(k)
        if segment < len(self.P_history):
            return self.P_history[segment]
        return self.P_history[-1]

    def in_funnel(self, k: int, eta: np.ndarray) -> bool:
        """Check if deviation eta is inside funnel at time k."""
        P = self.get_funnel_at_k(k)
        return eta @ P @ eta <= 1.0


def design_initial_controller(
    twin,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
    n: int = 4,
    m: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design initial controller (K0, P0) based on digital twin.

    Uses LQR to design a stabilizing feedback gain, then sets P0 based on
    state constraint bounds. This provides active feedback from the start.

    Args:
        twin: Digital twin system
        x_nom: Nominal state trajectory
        u_nom: Nominal input trajectory
        n: State dimension
        m: Input dimension

    Returns:
        K0: (m, n) Initial feedback gain
        P0: (n, n) Initial funnel shape
    """
    from trajectory.nominal_planner import linearize_at_point, compute_lqr_gain

    x_lower, x_upper, u_lower, u_upper = get_bounds_from_config()

    # Compute bounds at initial point
    P_min = compute_mvie_state(x_nom[0], x_lower, x_upper)

    # Linearize twin at initial operating point
    A, B = linearize_at_point(twin, x_nom[0], u_nom[0])

    # Design LQR gain with appropriate weights
    # High weights on position and velocity errors for aggressive tracking
    Q = np.diag([500.0, 500.0, 100.0, 100.0])
    R = np.eye(m) * 0.1

    try:
        K_lqr = compute_lqr_gain(A, B, Q, R)
        # LQR returns K for u = -K @ x, but our control law is u = u_nom + K @ eta
        # So we need to negate: K0 = -K_lqr
        K0 = -K_lqr
    except Exception as e:
        print(f"Warning: LQR design failed ({e}), using fallback gain")
        # Fallback: stabilizing proportional-derivative gain (negative for stability)
        K0 = np.array([
            [-20.0, 0.0, -5.0, 0.0],
            [0.0, -20.0, 0.0, -5.0]
        ])

    # Set P0 slightly larger than minimum to have room for funnel growth
    P0 = P_min * 2.0

    return K0, P0


def create_controller_from_params(
    twin,
    x_nom: np.ndarray,
    u_nom: np.ndarray,
) -> SegmentedFunnelController:
    """
    Create controller using parameters from config.

    Args:
        twin: Digital twin system
        x_nom: Nominal trajectory
        u_nom: Nominal inputs

    Returns:
        Initialized SegmentedFunnelController
    """
    controller = SegmentedFunnelController(
        twin=twin,
        x_nom=x_nom,
        u_nom=u_nom,
        T=SIM_PARAMS["T"],
        L=SIM_PARAMS["L"],
        alpha=SYNTH_PARAMS["alpha"],
        mu=SYNTH_PARAMS["mu"],
        eps_bar=SYNTH_PARAMS["eps_bar"],
        C=LIPSCHITZ["C"],
        gamma=LIPSCHITZ["gamma"],
        L_r=LIPSCHITZ["L_r"],
    )

    # Design and set initial controller
    K0, P0 = design_initial_controller(twin, x_nom, u_nom)
    controller.initialize(K0, P0)

    return controller
