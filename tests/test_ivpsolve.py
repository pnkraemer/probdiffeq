"""Tests for IVP solvers."""

import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    implementations,
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
    return backends.DynamicFilter(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_smoother(num_derivatives, information_op):
    return backends.DynamicSmoother(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize("information_op", [information.EK1(ode_dimension=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_ek1_filter(num_derivatives, information_op):
    return backends.DynamicFilter(
        implementation=implementations.DenseImplementation.from_num_derivatives(
            num_derivatives=num_derivatives, ode_dimension=1
        ),
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
        error_order=backend.implementation.num_derivatives + 1,
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
