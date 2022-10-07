"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    information,
    inits,
    problems,
    solvers,
    stepping,
)


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize("num_derivatives", [2])
@pytest_cases.parametrize("information_fn", [information.linearize_ek0_kron_1st])
def case_solver_adaptive_ek0(derivative_init_fn, num_derivatives, information_fn):
    odefilter = stepping.ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=backends.ekf0_isotropic_dynamic(
            num_derivatives=num_derivatives,
            information_fn=information_fn,
        ),
    )

    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=num_derivatives + 1,
    )


def case_ivp_logistic():
    def vf(x, t):
        return x * (1 - x)

    ivp_problem = problems.InitialValueProblem(
        vector_field=vf, initial_values=0.4, t0=0.0, t1=2.0, parameters=()
    )
    return ivp_problem


@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
@pytest_cases.parametrize_with_cases("ivp_problem", cases=".", prefix="case_ivp_")
def test_solver(solver, ivp_problem):
    assert isinstance(solver, solvers.AbstractIVPSolver)

    state = solver.init_fn(ivp=ivp_problem)
    assert state.dt_proposed > 0.0
    assert state.t == ivp_problem.t0
    assert jnp.shape(state.u) == jnp.shape(ivp_problem.initial_values)

    dt0 = 1.0
    state = solver.step_fn(state=state, vector_field=ivp_problem.vector_field, dt0=dt0)
    assert state.dt_proposed > 0.0
    assert state.t > ivp_problem.t0
    assert jnp.shape(state.u) == jnp.shape(ivp_problem.initial_values)
