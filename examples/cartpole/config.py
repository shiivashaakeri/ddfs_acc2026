"""Configuration for the cart-pole case study.

Two scenarios are supported via `TASK`:
  - "stabilize": pole near upright (θ_0 ≈ 0.3 rad), drive to rest at upright.
                 LQR rollout suffices; iLQR not needed.
  - "swing_up":  pole hanging (θ_0 = π), drive to upright. Requires iLQR
                 with energy-pumping; harder.
"""

from __future__ import annotations

import numpy as np

from dynamics.discretization import DiscreteSystem

from .dynamics import CartPole

PLANT_PARAMS = dict(M=1.00, m=0.20, l=0.50, b_c=0.05, b_p=0.005, g=9.81)
TWIN_PARAMS = dict(M=0.95, m=0.22, l=0.52, b_c=0.04, b_p=0.006, g=9.81)

STATE_BOUNDS = dict(
    p=(-2.5, 2.5),
    p_dot=(-8.0, 8.0),
    theta=(-1.5, 1.5),
    theta_dot=(-10.0, 10.0),
)
INPUT_BOUNDS = dict(force=(-25.0, 25.0))

SIM_PARAMS = dict(dt=0.01, N=600, T=100, L=60)
SYNTH_PARAMS = dict(alpha=0.98, mu=1.02, eps_bar=2.0)  # large excitation for richer data
LIPSCHITZ = dict(gamma=0.03, L_J=8.0, L_r=0.05, C=0.5, T_tilde=199)

# Default: stabilisation. The swing-up task widens the theta bounds and uses
# different initial states; see TASK_SWING_UP below.
TASK = "stabilize"

INITIAL_STATES = dict(
    x_nom_0=[0.10, 0.0, 0.30, 0.0],   # cart slightly off centre, pole 17° off upright
    x_goal=[0.0, 0.0, 0.0, 0.0],
    u_goal=[0.0],
    x_plant_0=[0.12, 0.05, 0.35, 0.10],  # plant starts slightly farther (||η_0|| ≈ 0.08)
)

# Reserved for swing-up: theta wrap-around is allowed (-2π to 2π).
TASK_SWING_UP = dict(
    x_nom_0=[0.0, 0.0, np.pi, 0.0],
    x_goal=[0.0, 0.0, 0.0, 0.0],
    u_goal=[0.0],
    x_plant_0=[0.05, 0.0, np.pi - 0.05, 0.05],
    STATE_BOUNDS=dict(
        p=(-2.5, 2.5),
        p_dot=(-10.0, 10.0),
        theta=(-4.0, 4.0),
        theta_dot=(-15.0, 15.0),
    ),
)


def create_plant() -> DiscreteSystem:
    cp = CartPole(**PLANT_PARAMS)
    return DiscreteSystem(cp.dynamics, SIM_PARAMS["dt"])


def create_twin() -> DiscreteSystem:
    cp = CartPole(**TWIN_PARAMS)
    return DiscreteSystem(cp.dynamics, SIM_PARAMS["dt"])
