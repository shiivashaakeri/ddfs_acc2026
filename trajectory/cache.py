"""Disk cache for the nominal trajectory (iLQR takes ~30s; we want it once)."""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

from .nominal_planner import NominalTrajectory, plan_nominal_with_twin_lqr

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _key(x0, x_goal, u_goal, N, system_id: str = "") -> str:
    h = hashlib.sha1()
    h.update(system_id.encode("utf-8"))
    for a in (x0, x_goal, u_goal):
        h.update(np.asarray(a, dtype=float).tobytes())
    h.update(int(N).to_bytes(8, "little"))
    return h.hexdigest()[:16]


def get_or_plan_nominal(
    x0: np.ndarray,
    x_goal: np.ndarray,
    u_goal: np.ndarray,
    N: int,
    u_lo: np.ndarray | None = None,
    u_hi: np.ndarray | None = None,
    force: bool = False,
    verbose: bool = False,
    system_id: str = "arm_2dof",   # identifier that participates in the cache key
    **kwargs,
) -> NominalTrajectory:
    _CACHE_DIR.mkdir(exist_ok=True)
    key = _key(x0, x_goal, u_goal, N, system_id)
    path = _CACHE_DIR / f"nominal_{system_id}_{key}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache hit] {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    if verbose:
        print(f"[cache miss] planning nominal -> {path}")
    nom = plan_nominal_with_twin_lqr(x0, x_goal, u_goal, N, u_lo=u_lo, u_hi=u_hi, **kwargs)
    with open(path, "wb") as f:
        pickle.dump(nom, f)
    return nom
