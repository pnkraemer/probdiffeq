"""Tests for IVP solvers."""


import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    information,
    inits,
    ivpsolve,
    problems,
    solvers,
    stepping,
)


@pytest_cases.case
def problem_logistic():
    return problems.InitialValueProblem(
        vector_field=lambda x, t: x * (1 - x),
        initial_values=0.5,
        t0=0.0,
        t1=10.0,
        parameters=(),
    )


@pytest_cases.parametrize("derivative_init_fn", [inits.taylor_mode, inits.forward_mode])
@pytest_cases.parametrize("controller", [controls.ProportionalIntegral()])
@pytest_cases.parametrize("information_fn", [information.linearize_ek0_kron_1st])
def solver_ek0(derivative_init_fn, controller, information_fn):
    odefilter = stepping.ODEFilter(
        derivative_init_fn=derivative_init_fn,
        backend=backends.ekf0_isotropic_dynamic(
            num_derivatives=2,
            information_fn=information_fn,
        ),
    )
    return solvers.Adaptive(
        stepping=odefilter,
        control=controller,
        atol=1e-5,
        rtol=1e-5,
        error_order=3,
    )


@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(ivp, solver):
    solution = ivpsolve.simulate_terminal_values(ivp, solver=solver)

    assert solution.t == ivp.t1
    assert jnp.allclose(solution.u, 1.0, atol=1e-1, rtol=1e-1)
