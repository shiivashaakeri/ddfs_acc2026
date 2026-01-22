"""
Persistence of Excitation Signals.

Generate bounded excitation signals to ensure data informativity:
- ||epsilon(k)|| <= eps_bar for all k in T_i^D
- Ensure rank([H_i; Xi_i]) = n + m
"""

# TODO: Implement excitation generation
# def generate_excitation(m, eps_bar) -> np.ndarray:
#     """
#     Generate random excitation signal.
#
#     Args:
#         m: input dimension (2)
#         eps_bar: bound on excitation norm (0.15)
#
#     Returns:
#         epsilon: (m,) excitation signal with ||epsilon|| <= eps_bar
#     """

# def verify_pe_condition(H, Xi, n, m) -> Tuple[bool, int]:
#     """
#     Verify persistence of excitation condition.
#
#     Returns:
#         satisfied: True if rank([H; Xi]) = n + m
#         rank: actual rank
#     """
