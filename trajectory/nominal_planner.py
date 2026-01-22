"""
Nominal Trajectory Planning.

Generate LQR-based nominal trajectory on the digital twin.

Approach:
1. Linearize twin at goal (x_goal, u_goal)
2. Compute infinite-horizon LQR gain K_lqr
3. Simulate twin from x_nom_0 using u = u_goal + K_lqr(x - x_goal)
4. Store {x_nom(k)}_{k=0}^N and {u_nom(k)}_{k=0}^{N-1}
"""

# TODO: Implement trajectory planning functions
# def linearize_at_point(system, x, u) -> Tuple[np.ndarray, np.ndarray]:
#     """Linearize system at (x, u) to get (A, B)."""

# def compute_lqr_gain(A, B, Q, R) -> np.ndarray:
#     """Compute discrete LQR gain using control.dlqr()."""

# def generate_nominal_trajectory(twin, x0, x_goal, u_goal, N, Q, R):
#     """Generate nominal trajectory on twin."""
#     # Returns: x_nom (N+1, 4), u_nom (N, 2)
