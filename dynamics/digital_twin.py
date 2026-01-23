"""
Digital Twin creation and mismatch computation.

Factory functions to create plant and twin systems with
mismatched parameters, and compute their mismatch.
"""

from __future__ import annotations

import numpy as np

from config.params import PLANT_PARAMS, SIM_PARAMS, TWIN_PARAMS
from dynamics.discretization import DiscreteSystem
from dynamics.robot_arm import RobotArm2DoF


def _build_arm(params: dict) -> RobotArm2DoF:
    return RobotArm2DoF(
        m1=params["m1"],
        m2=params["m2"],
        l1=params["l1"],
        l2=params["l2"],
        lc1=params["lc1"],
        lc2=params["lc2"],
        I1=params["I1"],
        I2=params["I2"],
        b1=params["b1"],
        b2=params["b2"],
        g=params.get("g", 9.81),
    )


def create_plant() -> DiscreteSystem:
    """Create the physical plant system."""
    plant_arm = _build_arm(PLANT_PARAMS)
    return DiscreteSystem(plant_arm.dynamics, SIM_PARAMS["dt"])


def create_twin() -> DiscreteSystem:
    """Create the digital twin system with mismatched params."""
    twin_arm = _build_arm(TWIN_PARAMS)
    return DiscreteSystem(twin_arm.dynamics, SIM_PARAMS["dt"])


def compute_mismatch(plant, twin, x, u) -> np.ndarray:
    """Compute one-step mismatch Delta(x,u) = x_next^plant - x_next^twin."""
    plant_next = plant.step(x, u)
    twin_next = twin.step(x, u)
    return np.asarray(plant_next) - np.asarray(twin_next)
