"""Tests for nominal trajectory planning."""

from __future__ import annotations

import numpy as np

from config.params import INITIAL_STATES, INPUT_BOUNDS, SIM_PARAMS, STATE_BOUNDS
from trajectory.nominal_planner import plan_nominal_with_twin_lqr


def _input_box():
    u_lo = np.array([INPUT_BOUNDS["tau1"][0], INPUT_BOUNDS["tau2"][0]])
    u_hi = np.array([INPUT_BOUNDS["tau1"][1], INPUT_BOUNDS["tau2"][1]])
    return u_lo, u_hi


def test_nominal_reaches_goal():
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    x_g = np.array(INITIAL_STATES["x_goal"])
    u_g = np.array(INITIAL_STATES["u_goal"])
    u_lo, u_hi = _input_box()
    nom = plan_nominal_with_twin_lqr(x0, x_g, u_g, SIM_PARAMS["N"], u_lo=u_lo, u_hi=u_hi, max_iter=80)
    assert np.linalg.norm(nom.x_nom[-1] - x_g) < 0.1


def test_nominal_respects_input_box():
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    x_g = np.array(INITIAL_STATES["x_goal"])
    u_g = np.array(INITIAL_STATES["u_goal"])
    u_lo, u_hi = _input_box()
    nom = plan_nominal_with_twin_lqr(x0, x_g, u_g, SIM_PARAMS["N"], u_lo=u_lo, u_hi=u_hi, max_iter=80)
    assert (nom.u_nom >= u_lo - 1e-9).all()
    assert (nom.u_nom <= u_hi + 1e-9).all()


def test_nominal_respects_state_box():
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    x_g = np.array(INITIAL_STATES["x_goal"])
    u_g = np.array(INITIAL_STATES["u_goal"])
    u_lo, u_hi = _input_box()
    nom = plan_nominal_with_twin_lqr(x0, x_g, u_g, SIM_PARAMS["N"], u_lo=u_lo, u_hi=u_hi, max_iter=80)
    for i, lab in enumerate(["q1", "q2", "dq1", "dq2"]):
        lo, hi = STATE_BOUNDS[lab]
        s = nom.x_nom[:, i]
        assert s.min() >= lo - 1e-6, f"{lab} below bound"
        assert s.max() <= hi + 1e-6, f"{lab} above bound"


def test_increment_bound_finite():
    x0 = np.array(INITIAL_STATES["x_nom_0"])
    x_g = np.array(INITIAL_STATES["x_goal"])
    u_g = np.array(INITIAL_STATES["u_goal"])
    u_lo, u_hi = _input_box()
    nom = plan_nominal_with_twin_lqr(x0, x_g, u_g, SIM_PARAMS["N"], u_lo=u_lo, u_hi=u_hi, max_iter=80)
    v = nom.increment_bound()
    assert 0 < v < 50.0
