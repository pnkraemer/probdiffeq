"""Tests for IVP solvers."""


import pytest_cases

from odefilter import ivp, ivpsolvers


@pytest_cases.case
def problem_logistic():
    return ivp.InitialValueProblem(
        f=lambda x: x * (1 - x),
        y0=0.5,
        t0=0.0,
        t1=1.0,
        p=(),
    )


@pytest_cases.case
def solver_ek0():
    return ivpsolvers.ek0(num_derivatives=2)


@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(problem, solver):

    solution = ivp.simulate_terminal_values(
        f=problem.f,
        tspan=(problem.t0, problem.t1),
        u0=problem.y0,
        solver=solver,
        atol=1e-4,
        rtol=1e-4,
    )

    assert solution is not None
