"""
Configuration parameters for DDFS ACC 2026 implementation.

This file contains all constants and parameters from the paper:
- Physical parameters (Table I): Plant and Twin parameters
- State/Input constraints
- Simulation parameters (dt, N, T, L)
- Synthesis parameters (alpha, mu, eps_bar)
- Lipschitz constants (gamma, L_J, L_r, C)
- Initial/Goal states
"""

PLANT_PARAMS = dict(
    m1=1.00,
    m2=0.80,
    l1=0.70,
    l2=0.60,
    lc1=0.35,
    lc2=0.30,
    I1=0.050,
    I2=0.040,
    b1=0.020,
    b2=0.020,
    g=9.81,
)

TWIN_PARAMS = dict(
    m1=0.95,
    m2=0.84,
    l1=0.73,
    l2=0.58,
    lc1=0.365,
    lc2=0.29,
    I1=0.055,
    I2=0.038,
    b1=0.018,
    b2=0.022,
    g=9.81,
)

STATE_BOUNDS = dict(
    q1=(-5.0, 9.0),
    q2=(-8.0, 8.0),
    dq1=(-8.0, 8.0),
    dq2=(-7.0, 7.0),
)

INPUT_BOUNDS = dict(tau1=(-40.0, 40.0), tau2=(-40.0, 40.0))

SIM_PARAMS = dict(dt=0.01, N=600, T=100, L=60)

SYNTH_PARAMS = dict(alpha=0.98, mu=1.02, eps_bar=0.15)

LIPSCHITZ = dict(gamma=0.034, L_J=16.94, L_r=0.074, C=3.64, T_tilde=199)

INITIAL_STATES = dict(
    x_nom_0=[0.28, -0.22, 0.0, 0.0],
    x_goal=[4.0, -1.0, 0.0, 0.0],
    u_goal=[-8.52, -2.37],
    x_plant_0=[2.28, 1.78, 1.0, -1.0],
)
