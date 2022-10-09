"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import controls, odefilters, solvers, taylor_series


@pytest_cases.parametrize(
    "tseries", [taylor_series.TaylorMode(), taylor_series.ForwardMode()]
)
@pytest_cases.parametrize_with_cases("backend", cases=".cases_backends")
def case_odefilter(tseries, backend):
    odefilter = odefilters.ODEFilter(
        taylor_series_init=tseries,
        backend=backend,
    )
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=backend.implementation.num_derivatives + 1,
    )


@pytest_cases.parametrize_with_cases("solver", cases=".")
@pytest_cases.parametrize_with_cases("vf, u0, t0, t1", cases=".cases_problems")
def test_solver(solver, vf, u0, t0, t1):
    assert isinstance(solver, solvers.AbstractIVPSolver)

    state0 = solver.init_fn(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
    )
    assert state0.dt_proposed > 0.0
    assert state0.accepted.t == t0
    assert jnp.shape(state0.accepted.u) == jnp.shape(u0[0])

    state1 = solver.step_fn(state=state0, vector_field=vf, t1=t1)
    assert isinstance(state0, type(state1))
    assert state1.dt_proposed > 0.0
    assert t0 < state1.accepted.t <= t1
    assert jnp.shape(state1.proposed.u) == jnp.shape(u0[0])
