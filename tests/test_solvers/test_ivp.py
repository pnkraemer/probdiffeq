"""Tests for IVP solvers."""

import jax
import jax.numpy as jnp
import pytest_cases

from odefilter import controls, inits, problems
from odefilter.solvers import ivp


@pytest_cases.parametrize("init", [inits.taylor_mode()])
@pytest_cases.parametrize("num_derivatives", [1])
def case_solver_ek0(init, num_derivatives):
    return ivp.ek0(init=init, num_derivatives=num_derivatives)


@pytest_cases.parametrize("init", [inits.taylor_mode()])
@pytest_cases.parametrize("num_derivatives", [1])
def case_solver_adaptive_ek0(init, num_derivatives):
    solver = ivp.ek0(init=init, num_derivatives=num_derivatives)
    control = controls.integral()
    atol, rtol = 1e-3, 1e-3
    return ivp.adaptive(
        solver=solver,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=num_derivatives + 1,
    )


def case_ivp_logistic():
    ode = problems.FirstOrderODE(f=lambda x: x * (1 - x))
    ivp_problem = problems.InitialValueProblem(
        ode_function=ode, initial_values=0.4, t0=0.0, t1=2.0
    )
    return ivp_problem


@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
@pytest_cases.parametrize_with_cases("ivp_problem", cases=".", prefix="case_ivp_")
def test_solver(solver, ivp_problem):
    alg, params = solver
    assert isinstance(alg, ivp.AbstractIVPSolver)

    state = alg.init_fn(ivp=ivp_problem, params=params)
    assert state.t == ivp_problem.t0
    assert jnp.shape(state.u) == jnp.shape(ivp_problem.initial_values)

    dt0 = 10.0
    state = alg.step_fn(
        state=state, ode_function=ivp_problem.ode_function, dt0=dt0, params=params
    )
    assert state.t > ivp_problem.t0
    assert jnp.shape(state.u) == jnp.shape(ivp_problem.initial_values)
