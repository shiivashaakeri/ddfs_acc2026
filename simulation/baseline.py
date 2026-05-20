"""Baseline simulation: apply the twin's nominal controller directly to the plant.

This is the comparison curve in the paper's Fig. 4: open-loop application of
û_nom(k) to the (true) plant, no online correction. The plant deviates from the
nominal because of model mismatch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def run_twin_only_baseline(
    plant_step: Callable, x_plant_0: np.ndarray, u_nom: np.ndarray
) -> np.ndarray:
    """Apply u_nom open-loop on the plant. Returns x_plant: (N+1, n)."""
    N = u_nom.shape[0]
    x = np.zeros((N + 1, x_plant_0.size))
    x[0] = x_plant_0
    for k in range(N):
        x[k + 1] = plant_step(x[k], u_nom[k])
    return x
