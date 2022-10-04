"""Tests for IVP solvers."""


import jax.numpy as jnp
import pytest_cases

from odefilter import inits, ivpsolve, ivpsolvers, problems, step
from odefilter.prob import markov


@pytest_cases.case
def problem_logistic():
    return problems.InitialValueProblem(
        ode_function=problems.FirstOrderODE(lambda x: x * (1 - x)),
        y0=0.5,
        t0=0.0,
        t1=10.0,
        p=(),
    )


@pytest_cases.parametrize("init", [inits.taylor_mode(), inits.forwardmode_jvp()])
@pytest_cases.parametrize(
    "control", [step.proportional_integral(atol=1e-5, rtol=1e-7, error_order=3)]
)
def solver_ek0(init, control):
    return ivpsolvers.ek0(
        num_derivatives=2,
        control=control,
        init=init,
    )


@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(ivp, solver):
    solver_alg, solver_params = solver
    solution = ivpsolve.simulate_terminal_values(
        ivp, solver=solver_alg, solver_params=solver_params
    )
    assert solution.t == ivp.t1

    mean, _ = solution.u
    assert jnp.allclose(mean[0], 1.0, atol=1e-3, rtol=1e-5)
