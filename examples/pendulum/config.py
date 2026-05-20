"""Configuration for the inverted-pendulum case study."""

from __future__ import annotations

import numpy as np

from dynamics.discretization import DiscreteSystem

from .dynamics import Pendulum

# Mass, length, damping. Twin differs by small mismatch (5-10%) like the arm.
PLANT_PARAMS = dict(m=1.00, l=1.00, b=0.10, g=9.81)
TWIN_PARAMS = dict(m=0.95, l=1.05, b=0.08, g=9.81)

# Constraints
STATE_BOUNDS = dict(theta=(-2.5, 2.5), theta_dot=(-8.0, 8.0))
INPUT_BOUNDS = dict(tau=(-12.0, 12.0))

# Simulation / segmentation
SIM_PARAMS = dict(dt=0.01, N=600, T=100, L=60)

# Synthesis (same as the arm; the LMI itself is system-agnostic)
SYNTH_PARAMS = dict(alpha=0.98, mu=1.02, eps_bar=0.10)

# Lipschitz estimates -- placeholders; estimate empirically at runtime.
LIPSCHITZ = dict(gamma=0.02, L_J=8.0, L_r=0.05, C=0.5, T_tilde=199)

# Stabilize from a moderate offset to the upright. Initial deviation is the
# plant being slightly farther off than the nominal starts.
INITIAL_STATES = dict(
    x_nom_0=[0.6, 0.0],         # ~34° off upright
    x_goal=[0.0, 0.0],          # upright equilibrium
    u_goal=[0.0],
    x_plant_0=[0.75, 0.1],      # plant starts farther, with small velocity
)


def create_plant() -> DiscreteSystem:
    pend = Pendulum(**PLANT_PARAMS)
    return DiscreteSystem(pend.dynamics, SIM_PARAMS["dt"])


def create_twin() -> DiscreteSystem:
    pend = Pendulum(**TWIN_PARAMS)
    return DiscreteSystem(pend.dynamics, SIM_PARAMS["dt"])
