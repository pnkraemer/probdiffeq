"""Solution cases."""

import jax
import jax.numpy as jnp
import pytest_cases
from diffeqzoo import ivps
from jax.experimental.ode import odeint

from odefilter import ivpsolve


@pytest_cases.fixture(scope="session", name="ode_problem")
def fixture_ode_problem():
    f, u0, (t0, t1), f_args = ivps.lotka_volterra()
    t1 = 2.0

    @jax.jit
    def vf(x, *, t, p):
        return f(x, *p)

    # Only very short time-intervals are sufficient for a unit test.
    return vf, (u0,), t0, t1, f_args


@pytest_cases.fixture(scope="session", name="tolerances")
def fixture_tolerances():
    return 1e-5, 1e-3


@pytest_cases.fixture(scope="session", name="reference_terminal_values")
def fixture_reference_terminal_values(ode_problem, tolerances):
    vf, (u0,), t0, t1, f_args = ode_problem
    atol, rtol = tolerances

    @jax.jit
    def func(y, t, *p):
        return vf(y, t=t, p=p)

    ts = jnp.asarray([t0, t1])
    odeint_solution = odeint(func, u0, ts, *f_args, atol=atol, rtol=rtol)
    ys_reference = odeint_solution[-1, :]
    return t1, ys_reference


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases")
def fixture_solution_terminal_values(ode_problem, tolerances, solver):
    vf, u0, t0, t1, f_args = ode_problem
    atol, rtol = tolerances

    solution = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=f_args, solver=solver, atol=atol, rtol=rtol
    )
    return solution.t, solution.u
