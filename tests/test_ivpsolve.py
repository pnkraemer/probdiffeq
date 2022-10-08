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
    solvers,
)


@pytest_cases.case
def case_problem_logistic():
    return lambda x, t: x * (1 - x), (jnp.asarray([0.5]),), 0.0, 10.0, ()


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_filter(num_derivatives, information_op):
    return backends.DynamicIsotropicFilter.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_smoother(num_derivatives, information_op):
    return backends.DynamicIsotropicSmoother.from_num_derivatives(
        num_derivatives=num_derivatives,
        information=information_op,
    )


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize_with_cases("ek0", cases=".", prefix="case_backend_")
def case_solver_odefilter(derivative_init_fn, ek0):
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


@pytest_cases.parametrize_with_cases(
    "vf, u0, t0, t1, p", cases=".", prefix="case_problem_"
)
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
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
