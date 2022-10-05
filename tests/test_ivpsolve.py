"""Tests for IVP solvers."""


import jax.numpy as jnp
import pytest_cases

from odefilter import controls, inits, ivpsolve, problems
from odefilter.solvers import ivp


@pytest_cases.case
def problem_logistic():
    return problems.InitialValueProblem(
        ode_function=problems.FirstOrderODE(lambda x: x * (1 - x)),
        initial_values=0.5,
        t0=0.0,
        t1=10.0,
        parameters=(),
    )


@pytest_cases.parametrize("init", [inits.taylor_mode(), inits.forwardmode_jvp()])
@pytest_cases.parametrize("controller", [controls.proportional_integral()])
def solver_ek0(init, controller):
    solver = ivp.ek0_non_adaptive(
        num_derivatives=2,
        init=init,
    )
    return ivp.adaptive(
        non_adaptive_solver=solver,
        control=controller,
        atol=1e-5,
        rtol=1e-5,
        error_order=3,
    )


@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(ivp, solver):
    solver_alg, solver_params = solver
    solution = ivpsolve.simulate_terminal_values(
        ivp, solver=solver_alg, solver_params=solver_params
    )

    assert solution.t == ivp.t1
    assert jnp.allclose(solution.u, 1.0, atol=1e-1, rtol=1e-1)
