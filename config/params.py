"""
Configuration parameters for DDFS ACC 2026 implementation.

This file contains all constants and parameters from the paper:
"Robust Data-Driven Control for Nonlinear Systems Using their
Digital Twins and Quadratic Funnels"

All values are extracted from Table I and Section VI of the paper.
"""

import numpy as np

# =============================================================================
# Physical Parameters (Table I)
# =============================================================================

# Plant parameters (true physical system)
PLANT_PARAMS = dict(
    m1=1.00,      # Link 1 mass [kg]
    m2=0.80,      # Link 2 mass [kg]
    l1=0.70,      # Link 1 length [m]
    l2=0.60,      # Link 2 length [m]
    lc1=0.35,     # Link 1 center of mass distance [m]
    lc2=0.30,     # Link 2 center of mass distance [m]
    I1=0.050,     # Link 1 inertia [kg·m²]
    I2=0.040,     # Link 2 inertia [kg·m²]
    b1=0.020,     # Link 1 viscous friction [N·m·s/rad]
    b2=0.020,     # Link 2 viscous friction [N·m·s/rad]
    g=9.81        # Gravity [m/s²]
)

# Twin parameters (digital twin with mismatched parameters)
TWIN_PARAMS = dict(
    m1=0.95,      # Link 1 mass [kg]
    m2=0.84,      # Link 2 mass [kg]
    l1=0.73,      # Link 1 length [m]
    l2=0.58,      # Link 2 length [m]
    lc1=0.365,    # Link 1 center of mass distance [m]
    lc2=0.29,     # Link 2 center of mass distance [m]
    I1=0.055,     # Link 1 inertia [kg·m²]
    I2=0.038,     # Link 2 inertia [kg·m²]
    b1=0.018,     # Link 1 viscous friction [N·m·s/rad]
    b2=0.022,     # Link 2 viscous friction [N·m·s/rad]
    g=9.81        # Gravity [m/s²]
)

# =============================================================================
# State and Input Constraints
# =============================================================================

# State bounds: x = [q1, q2, q1_dot, q2_dot]
STATE_BOUNDS = dict(
    q1=(-5.0, 9.0),       # Joint 1 angle [rad]
    q2=(-8.0, 8.0),       # Joint 2 angle [rad]
    q1_dot=(-8.0, 8.0),   # Joint 1 velocity [rad/s]
    q2_dot=(-7.0, 7.0)    # Joint 2 velocity [rad/s]
)

# Input bounds: u = [tau1, tau2]
INPUT_BOUNDS = dict(
    tau1=(-40.0, 40.0),   # Joint 1 torque [N·m]
    tau2=(-40.0, 40.0)    # Joint 2 torque [N·m]
)

# Convenient array forms for clipping
STATE_LOWER = np.array([-5.0, -8.0, -8.0, -7.0])
STATE_UPPER = np.array([9.0, 8.0, 8.0, 7.0])
INPUT_LOWER = np.array([-40.0, -40.0])
INPUT_UPPER = np.array([40.0, 40.0])

# =============================================================================
# Simulation Parameters
# =============================================================================

SIM_PARAMS = dict(
    dt=0.01,      # Discretization time step [s]
    N=600,        # Total number of steps (6 seconds)
    T=100,        # Segment length [steps] (1 second per segment)
    L=60          # Data window length [steps]
)

# Derived parameters
NUM_SEGMENTS = SIM_PARAMS['N'] // SIM_PARAMS['T']  # = 6 segments

# =============================================================================
# Synthesis Parameters
# =============================================================================

SYNTH_PARAMS = dict(
    alpha=0.98,       # Lyapunov decay rate (0 < α < 1)
    mu=1.02,          # Cross-segment growth bound (P_{i+1} ≤ μ P_i)
    eps_bar=0.15      # Excitation signal bound ||ε(k)|| ≤ ε̄
)

# =============================================================================
# Lipschitz Constants (estimated from paper, Section VI)
# =============================================================================

LIPSCHITZ = dict(
    gamma=0.034,      # Mismatch bound: ||Δ(x,u)|| ≤ γ
    L_J=16.94,        # Jacobian Lipschitz constant
    L_r=0.074,        # Second-order remainder constant
    v=0.215,          # Trajectory speed (max increment norm)
    C=3.64,           # Jacobian variation bound C = L_J * v
    T_tilde=199       # Time horizon for variation bound (2T - 1)
)

# =============================================================================
# Initial and Goal States
# =============================================================================

INITIAL_STATES = dict(
    # Nominal trajectory start (on digital twin)
    x_nom_0=np.array([0.28, -0.22, 0.0, 0.0]),

    # Goal state
    x_goal=np.array([4.0, -1.0, 0.0, 0.0]),

    # Goal input (equilibrium torques at goal)
    u_goal=np.array([-8.52, -2.37]),

    # Plant initial state (perturbed from nominal)
    x_plant_0=np.array([2.28, 1.78, 1.0, -1.0])
)

# =============================================================================
# MVIE Initial Values (from paper, at k=0)
# =============================================================================

# P_min^{1/2}(0) = diag(0.19, 0.12, 0.12, 0.14)
P_MIN_SQRT_0 = np.diag([0.19, 0.12, 0.12, 0.14])
P_MIN_0 = P_MIN_SQRT_0 @ P_MIN_SQRT_0  # P_min(0)

# R_max^{1/2}(0) = diag(28.56, 37.62)
R_MAX_SQRT_0 = np.diag([28.56, 37.62])
R_MAX_0 = R_MAX_SQRT_0 @ R_MAX_SQRT_0  # R_max(0)

# =============================================================================
# LQR Weights for Nominal Trajectory Generation
# =============================================================================

LQR_WEIGHTS = dict(
    Q=np.diag([10.0, 10.0, 1.0, 1.0]),   # State cost
    R=np.diag([0.1, 0.1])                 # Input cost
)

# =============================================================================
# System Dimensions
# =============================================================================

STATE_DIM = 4  # n = 4 (q1, q2, q1_dot, q2_dot)
INPUT_DIM = 2  # m = 2 (tau1, tau2)
