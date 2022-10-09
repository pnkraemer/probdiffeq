"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases
from jax.experimental.ode import odeint

from odefilter import controls, ivpsolve, odefilters, solvers, taylor_series


@pytest_cases.parametrize(
    "taylor",
    [taylor_series.TaylorMode(), taylor_series.ForwardMode()],
    ids=["TaylorMode", "ForwardMode"],
)
@pytest_cases.parametrize_with_cases("backend", cases=".cases_backends")
def case_odefilter(taylor, backend):
    odefilter = odefilters.ODEFilter(
        taylor_series_init=taylor,
        backend=backend,
    )
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-5, 1e-5
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=backend.implementation.num_derivatives + 1,
    )


@pytest_cases.parametrize_with_cases("vf, u0, t0, t1, p", cases=".cases_problems")
@pytest_cases.parametrize_with_cases("solver", cases=".")
def test_simulate_terminal_values(vf, u0, t0, t1, p, solver):
    odeint_solution = odeint(vf, u0[0], jnp.asarray([t0, t1]), *p, atol=1e-6, rtol=1e-6)
    ys_reference = odeint_solution[-1, :]

    solution = ivpsolve.simulate_terminal_values(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
    )

    assert solution.t == t1
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)


@pytest_cases.parametrize_with_cases("vf, u0, t0, t1, p", cases=".cases_problems")
@pytest_cases.parametrize_with_cases("solver", cases=".")
def test_simulate_checkpoints(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=10)

    odeint_solution = odeint(vf, u0[0], ts, *p, atol=1e-6, rtol=1e-6)
    ts_reference, ys_reference = ts[1:], odeint_solution[1:, :]

    solution = ivpsolve.simulate_checkpoints(
        vector_field=vf,
        initial_values=u0,
        ts=ts,
        parameters=p,
        solver=solver,
    )
    assert jnp.allclose(solution.t, ts_reference)
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)
