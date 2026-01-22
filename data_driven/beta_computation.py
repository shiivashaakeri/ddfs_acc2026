"""
Beta (disturbance energy bound) Computation.

Compute beta_i from Eq. 16:
beta_i = sum_{k in T_i^D} (C|k-k_i| * ||[eta(k); xi(k)]|| + gamma + L_r * ||[eta(k); xi(k)]||^2)^2
"""

# TODO: Implement beta computation
# def compute_beta(eta_log, xi_log, data_window, k_i, C, gamma, L_r) -> float:
#     """
#     Compute beta_i from Eq. 16.
#
#     Args:
#         eta_log: dict of eta(k) values
#         xi_log: dict of xi(k) values
#         data_window: indices in T_i^D
#         k_i: start of segment i
#         C, gamma, L_r: Lipschitz constants
#
#     Returns:
#         beta_i: scalar bound
#     """
#     beta_i = 0.0
#     for k in data_window:
#         eta_k = eta_log[k]
#         xi_k = xi_log[k]
#         z_k = np.concatenate([eta_k, xi_k])
#         norm_z = np.linalg.norm(z_k)
#         term = C * abs(k - k_i) * norm_z + gamma + L_r * norm_z**2
#         beta_i += term**2
#     return beta_i
