"""
Ellipsoid Bounds Computation (MVIE).

Compute P_min(k), R_max(k) via Maximum Volume Inscribed Ellipsoid (Remark 2).

Paper values at k=0:
    P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14)
    R_max^{1/2}(0) = diag(28.56, 37.62)
"""

# TODO: Implement MVIE computation
# def compute_mvie_state(x_nom_k, x_bounds) -> np.ndarray:
#     """
#     Compute P_min(k) for state constraints.
#
#     Args:
#         x_nom_k: nominal state at time k (4,)
#         x_bounds: dict with 'lower' and 'upper' arrays (4,) each
#
#     Returns:
#         P_min_k: (4, 4) positive definite matrix
#     """

# def compute_mvie_input(u_nom_k, u_bounds) -> np.ndarray:
#     """
#     Compute R_max(k) for input constraints.
#
#     Args:
#         u_nom_k: nominal input at time k (2,)
#         u_bounds: dict with 'lower' and 'upper' arrays (2,) each
#
#     Returns:
#         R_max_k: (2, 2) positive semidefinite matrix
#     """

# def compute_segment_bounds(x_nom, u_nom, segment_indices, x_bounds, u_bounds):
#     """Compute P_min_i, R_max_i for segment i."""
