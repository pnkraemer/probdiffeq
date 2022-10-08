"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    information,
    inits,
    ivpsolve,
    odefilters,
    problems,
    solvers,
)


@pytest_cases.case
def case_problem_logistic():
    return problems.InitialValueProblem(
        vector_field=lambda x, t: x * (1 - x),
        initial_values=0.5,
        t0=0.0,
        t1=10.0,
        parameters=(),
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
def case_ek0_smoother(num_derivatives, information_op):
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
    atol, rtol = 1e-5, 1e-5
    return solvers.Adaptive(
        stepping=odefilter,
        control=control,
        atol=atol,
        rtol=rtol,
        error_order=ek0.num_derivatives + 1,
    )


@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="case_problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
def test_simulate_terminal_values(ivp, solver):
    solution = ivpsolve.simulate_terminal_values(ivp, solver=solver)

    assert solution.t == ivp.t1
    assert jnp.allclose(solution.u, 1.0, atol=1e-1, rtol=1e-1)
