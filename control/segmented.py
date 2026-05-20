"""Algorithm 1: online segmented data-driven funnel control.

For i = 0, 1, ... while k_i < N:
    1. Apply fixed K_i over T_i = {k_i, ..., k_{i+1}-1}.
       During T_i^D = {k_{i+1}-L, ..., k_{i+1}-1}, add excitation ε(k).
    2. Log (η_k, ξ_k, η_{k+1}).
    3. At k_{i+1}: assemble (H_i, H_i^+, Ξ_i), compute β_i, build the
       SegmentUncertainty bundle, and solve Eq. (22) to get (P_{i+1}, K_{i+1}).

Fixed K per segment. No gain ramps, no LQR fallback, no β capping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from data_driven.beta import beta_empirical, beta_segment
from data_driven.data_collection import (
    BoundedExcitation,
    SegmentData,
    assemble_data_matrices,
    persistently_exciting,
)
from data_driven.uncertainty import SegmentUncertainty
from synthesis.sdp import FunnelDesign, synthesize_funnel


@dataclass
class SegmentLog:
    i: int
    k_start: int
    k_end: int  # exclusive
    P: np.ndarray
    K: np.ndarray
    eta: np.ndarray            # (T, n)  -- η for k in T_i
    xi: np.ndarray             # (T, m)  -- ξ for k in T_i
    eta_next: np.ndarray       # (T, n)  -- η for k+1 in T_i
    x_plant: np.ndarray        # (T, n)  -- absolute plant state
    u_plant: np.ndarray        # (T, m)  -- absolute plant input
    beta: float | None = None
    persistently_exciting: bool | None = None


@dataclass
class AlgorithmOutputs:
    segments: list[SegmentLog] = field(default_factory=list)
    x_plant_full: np.ndarray | None = None
    sdp_designs: list[FunnelDesign] = field(default_factory=list)


def run_algorithm_1(
    plant_step: Callable,
    x_plant_0: np.ndarray,
    x_nom: np.ndarray,         # (N+1, n)
    u_nom: np.ndarray,         # (N, m)
    P_max_shape_per_segment: list[np.ndarray],   # shape-matrix upper bound per segment
    R_max_per_segment: list[np.ndarray],
    P0: np.ndarray,
    K0: np.ndarray,
    T: int,
    L: int,
    alpha: float,
    mu: float,
    eps_bar: float,
    gamma: float,
    C: float,
    L_r: float,
    T_tilde: float,
    CT_per_segment: list[float] | None = None,   # if provided, C·T̃ override per segment
    u_lo: np.ndarray | None = None,
    u_hi: np.ndarray | None = None,
    beta_mode: str = "paper",        # "paper" (Eq. 16) | "empirical" (LS-tight) | "ce" (certainty-equivalence)
    beta_safety: float = 4.0,
    ce_beta: float = 1e-10,
    rng_seed: int = 0,
    verbose: bool = False,
) -> AlgorithmOutputs:
    """Run Algorithm 1 end-to-end on the plant.

    The horizon is N = u_nom.shape[0]. The number of segments is
    S = ceil(N / T). The last segment is truncated if N is not a multiple of T.
    """
    N = u_nom.shape[0]
    n = x_nom.shape[1]
    m = u_nom.shape[1]
    n_segments = (N + T - 1) // T
    assert len(P_max_shape_per_segment) >= n_segments
    assert len(R_max_per_segment) >= n_segments

    rng = np.random.default_rng(rng_seed)
    excitation = BoundedExcitation(eps_bar=eps_bar, dim=m, rng=rng)

    outputs = AlgorithmOutputs()
    x_plant_full = np.zeros((N + 1, n))
    x_plant_full[0] = x_plant_0

    P_i = P0.copy()
    K_i = K0.copy()

    for i in range(n_segments):
        k_start = i * T
        k_end = min(k_start + T, N)        # exclusive; len = k_end - k_start
        seg_len = k_end - k_start
        data_window_start = max(k_end - L, k_start)  # k_i^D
        data_window_len = k_end - data_window_start

        # ---- Execute segment T_i.
        seg_eta = np.zeros((seg_len, n))
        seg_xi = np.zeros((seg_len, m))
        seg_eta_next = np.zeros((seg_len, n))
        seg_xplant = np.zeros((seg_len, n))
        seg_uplant = np.zeros((seg_len, m))
        for j in range(seg_len):
            k = k_start + j
            x_k = x_plant_full[k]
            eta_k = x_k - x_nom[k]
            xi_k = K_i @ eta_k
            if k >= data_window_start:
                xi_k = xi_k + excitation.sample()
            u_k = u_nom[k] + xi_k
            if u_lo is not None and u_hi is not None:
                u_clipped = np.clip(u_k, u_lo, u_hi)
                # Reconstruct the actual ξ that was applied (post-clip) so the
                # logged data matches what the plant saw.
                xi_k = u_clipped - u_nom[k]
                u_k = u_clipped
            x_next = plant_step(x_k, u_k)
            x_plant_full[k + 1] = x_next
            seg_eta[j] = eta_k
            seg_xi[j] = xi_k
            seg_eta_next[j] = x_next - x_nom[k + 1]
            seg_xplant[j] = x_k
            seg_uplant[j] = u_k

        seg_log = SegmentLog(
            i=i, k_start=k_start, k_end=k_end, P=P_i, K=K_i,
            eta=seg_eta, xi=seg_xi, eta_next=seg_eta_next,
            x_plant=seg_xplant, u_plant=seg_uplant,
        )

        # ---- Assemble data window.
        seg_data = SegmentData(
            eta=seg_eta[-data_window_len:],
            xi=seg_xi[-data_window_len:],
            eta_next=seg_eta_next[-data_window_len:],
            k_start=data_window_start,
            k_segment_start=k_start,
        )
        H, H_plus, Xi = assemble_data_matrices(seg_data)
        pe = persistently_exciting(H, Xi)
        if beta_mode == "paper":
            beta_i = beta_segment(seg_data, C=C, gamma=gamma, L_r=L_r)
        elif beta_mode == "empirical":
            beta_i = beta_empirical(H, H_plus, Xi, safety=beta_safety)
        elif beta_mode == "ce":
            # Certainty-equivalence: replace real (H, H+, Xi) with a clean synthetic
            # probe that pins (A, B) = (A_LS, B_LS) exactly. The LMI then designs a
            # certified controller for the LS-estimated system.
            stack = np.vstack([H, Xi])
            AB_ls = H_plus @ stack.T @ np.linalg.pinv(stack @ stack.T)
            A_ls, B_ls = AB_ls[:, :n], AB_ls[:, n:]
            n_synth = n + m
            H = np.zeros((n, n_synth)); H[:, :n] = np.eye(n)
            Xi = np.zeros((m, n_synth)); Xi[:, n:] = np.eye(m)
            H_plus = A_ls @ H + B_ls @ Xi
            beta_i = ce_beta
        else:
            raise ValueError(f"unknown beta_mode: {beta_mode}")
        seg_log.beta = beta_i
        seg_log.persistently_exciting = pe
        outputs.segments.append(seg_log)

        if verbose:
            print(
                f"[seg {i}] k=[{k_start}, {k_end}), "
                f"||η_end||={np.linalg.norm(seg_eta[-1]):.4f}, "
                f"β={beta_i:.3e}, PE={pe}"
            )

        # ---- Design next segment.
        if i + 1 >= n_segments:
            break  # last segment; no more design needed

        if CT_per_segment is not None:
            # Empirical override: bundle uses C=CT_value, T_tilde=1 so the product
            # C * T_tilde matches the empirical Jacobian variation.
            bundle = SegmentUncertainty(
                H=H, H_plus=H_plus, Xi=Xi,
                beta=beta_i, C=CT_per_segment[i], T_tilde=1.0,
            )
        else:
            bundle = SegmentUncertainty(
                H=H, H_plus=H_plus, Xi=Xi,
                beta=beta_i, C=C, T_tilde=T_tilde,
            )
        P_max_next = P_max_shape_per_segment[i + 1]
        R_max_next = R_max_per_segment[i + 1]
        design = synthesize_funnel(
            bundle,
            P_max_shape=P_max_next,
            R_max=R_max_next,
            alpha=alpha,
            P_prev=None,    # cross-segment growth disabled (see SDP module note)
            mu=None,
        )
        outputs.sdp_designs.append(design)
        P_i = design.P
        K_i = design.K

    outputs.x_plant_full = x_plant_full
    return outputs
