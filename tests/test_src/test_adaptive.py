"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import adaptive, controls, odefilters, taylor_series


@pytest_cases.parametrize_with_cases("strategy", cases=".cases_strategies")
def case_odefilter(strategy):
    odefilter = odefilters.ODEFilter(strategy=strategy)
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return adaptive.Adaptive(
        odefilter=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=strategy.implementation.num_derivatives + 1,
    )


@pytest_cases.parametrize_with_cases("solver", cases=".")
@pytest_cases.parametrize_with_cases("vf, u0, t0, t1, p", cases=".cases_problems")
def test_solver(solver, vf, u0, t0, t1, p):
    assert isinstance(solver, adaptive.Adaptive)

    def vf_p(*ys, t):
        return vf(*ys, t, *p)

    def vf_p_0(*ys):
        return vf_p(*ys, t=t0)

    taylor_series_fn = taylor_series.TaylorMode()
    tcoeffs = taylor_series_fn(
        vector_field=vf_p_0,
        initial_values=u0,
        num=solver.odefilter.strategy.implementation.num_derivatives,
    )
    state0 = solver.init_fn(
        taylor_coefficients=tcoeffs,
        t0=t0,
    )
    assert state0.dt_proposed > 0.0
    assert state0.accepted.t == t0
    assert jnp.shape(state0.accepted.u) == jnp.shape(u0[0])

    state1 = solver.step_fn(state=state0, vector_field=vf_p, t1=t1)
    assert isinstance(state0, type(state1))
    assert state1.dt_proposed > 0.0
    assert t0 < state1.accepted.t <= t1
    assert jnp.shape(state1.proposed.u) == jnp.shape(u0[0])
