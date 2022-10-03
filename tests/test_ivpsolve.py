"""Tests for IVP solvers."""


import jax.numpy as jnp
import pytest_cases

from odefilter import ivp, ivpsolvers, markov


@pytest_cases.case
def problem_logistic():
    return ivp.InitialValueProblem(
        f=lambda x: x * (1 - x),
        y0=0.5,
        t0=0.0,
        t1=10.0,
        p=(),
    )


@pytest_cases.case
def solver_ek0():
    return ivpsolvers.ek0(num_derivatives=2)


@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(problem, solver):

    t, solution = ivp.simulate_terminal_values(
        f=problem.f,
        tspan=(problem.t0, problem.t1),
        u0=problem.y0,
        solver=solver,
        atol=1e-5,
        rtol=1e-7,
    )
    mean, cov_sqrtm_upper = solution.u
    assert jnp.allclose(mean[0], 1.0, atol=1e-3, rtol=1e-5)
