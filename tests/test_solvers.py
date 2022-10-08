"""Tests for IVP solvers."""

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
def case_backend_ek0_filter(num_derivatives, information_op):
    return backends.DynamicIsotropicFilter.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize("information_op", [information.IsotropicEK0(ode_order=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_ek0_smoother(num_derivatives, information_op):
    return backends.DynamicIsotropicSmoother.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize_with_cases("backend", cases=".", prefix="case_backend_")
def case_solver_adaptive_ek0(derivative_init_fn, backend):
    odefilter = odefilters.ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=backend,
    )
    control = controls.ProportionalIntegral()
    atol, rtol = 1e-3, 1e-3
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=backend.num_derivatives + 1,
    )


def case_ivp_logistic():
    def vf(x, t):
        return x * (1 - x)

    ivp_problem = problems.InitialValueProblem(
        vector_field=vf, initial_values=(0.4,), t0=0.0, t1=2.0, parameters=()
    )
    return ivp_problem


@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
@pytest_cases.parametrize_with_cases("ivp_problem", cases=".", prefix="case_ivp_")
def test_solver(solver, ivp_problem):
    assert isinstance(solver, solvers.AbstractIVPSolver)

    state0 = solver.init_fn(
        vector_field=ivp_problem.vector_field,
        initial_values=ivp_problem.initial_values,
        t0=ivp_problem.t0,
    )
    assert state0.dt_proposed > 0.0
    assert state0.accepted.t == ivp_problem.t0
    assert jnp.shape(state0.accepted.u) == jnp.shape(ivp_problem.initial_values[0])

    state1 = solver.step_fn(
        state=state0, vector_field=ivp_problem.vector_field, t1=ivp_problem.t1
    )
    assert isinstance(state0, type(state1))
    assert state1.dt_proposed > 0.0
    assert ivp_problem.t0 < state1.accepted.t <= ivp_problem.t1
    assert jnp.shape(state1.proposed.u) == jnp.shape(ivp_problem.initial_values[0])
