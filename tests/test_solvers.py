"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    information,
    inits,
    odefilters,
    problems,
    solvers,
)


@pytest_cases.parametrize("information_op", [information.IsotropicEK0(ode_order=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_ek0_filter(num_derivatives, information_op):
    return backends.DynamicIsotropicFilter.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize("information_op", [information.IsotropicEK0(ode_order=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_ek0_filter(num_derivatives, information_op):
    return backends.DynamicIsotropicSmoother.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize_with_cases("ek0", cases=".", prefix="case_ek0_")
def case_solver_adaptive_ek0(derivative_init_fn, ek0):
    odefilter = odefilters.ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=ek0,
    )
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=ek0.num_derivatives + 1,
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

    state0 = solver.init_fn(ivp=ivp_problem)
    assert state0.dt_proposed > 0.0
    assert state0.stepping.t == ivp_problem.t0
    assert jnp.shape(state0.stepping.u) == jnp.shape(ivp_problem.initial_values)

    dt0 = 1.0
    state1 = solver.step_fn(
        state=state0, vector_field=ivp_problem.vector_field, dt0=dt0
    )
    assert isinstance(state0, type(state1))
    assert state1.dt_proposed > 0.0
    assert state1.stepping.t > ivp_problem.t0
    assert jnp.shape(state1.stepping.u) == jnp.shape(ivp_problem.initial_values)
