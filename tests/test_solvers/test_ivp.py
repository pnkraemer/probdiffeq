"""Tests for IVP solvers."""

import pytest_cases

from odefilter import controls, inits, problems
from odefilter.solvers import ivp


@pytest_cases.parametrize("init", [inits.taylor_mode()])
@pytest_cases.parametrize("num_derivatives", [1])
def case_solver_ek0(init, num_derivatives):
    return ivp.ek0(init=init, num_derivatives=num_derivatives)


def case_ivp_logistic():
    ode = problems.FirstOrderODE(f=lambda x: x * (1 - x))
    ivp_problem = problems.InitialValueProblem(
        ode_function=ode, initial_values=0.4, t0=0.0, t1=2.0
    )
    return ivp_problem


@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="case_solver_")
@pytest_cases.parametrize_with_cases("ivp_problem", cases=".", prefix="case_ivp_")
def test_solver(solver, ivp_problem):
    alg, params = solver
    assert isinstance(alg, ivp.AbstractIVPSolver)

    state = alg.init_fn(ivp=ivp_problem, params=params)
