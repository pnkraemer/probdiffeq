"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import controls, inits, ivpsolve, odefilters, solvers


@pytest_cases.parametrize("init_fn", [inits.TaylorMode(), inits.ForwardMode()])
@pytest_cases.parametrize_with_cases("backend", cases=".cases_backends")
def case_solver_odefilter(init_fn, backend):
    odefilter = odefilters.ODEFilter(
        derivative_init_fn=init_fn,
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


@pytest_cases.parametrize_with_cases("vf, u0, t0, t1, p", cases=".cases_problems")
@pytest_cases.parametrize_with_cases("solver", cases=".")
def test_simulate_terminal_values(vf, u0, t0, t1, p, solver):
    solution = ivpsolve.simulate_terminal_values(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
    )

    assert solution.t == t1
    assert jnp.allclose(solution.u, 1.0, atol=1e-1, rtol=1e-1)
