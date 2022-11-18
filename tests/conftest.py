"""Test configurations."""

import diffeqzoo.ivps
import jax
import jax.experimental.ode
import jax.numpy as jnp
import pytest_cases
import pytest_cases.filters

from probdiffeq import ivpsolve, taylor

# Set some test filters

_IS_FILTER = pytest_cases.filters.has_tag("filter")
_IS_SMOOTHER = pytest_cases.filters.has_tag("smoother")
_IS_FIXEDPOINT = pytest_cases.filters.has_tag("fixedpoint")

# terminal value simulation uses all solvers
_CHECKPT = _IS_FILTER | _IS_FIXEDPOINT
_SOLVE = _IS_FILTER | _IS_SMOOTHER

# Common fixtures (ODE problem, tolerances)


@pytest_cases.fixture(scope="session", name="ode_problem")
def fixture_ode_problem():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    # Only very short time-intervals are sufficient for a unit test.
    return vf, (u0,), t0, t1, f_args


@pytest_cases.fixture(scope="session", name="tolerances")
def fixture_tolerances():
    return 1e-5, 1e-3


# Terminal value fixtures


@pytest_cases.fixture(scope="session", name="reference_terminal_values")
def fixture_reference_terminal_values(ode_problem, tolerances):
    vf, (u0,), t0, t1, f_args = ode_problem
    atol, rtol = tolerances

    @jax.jit
    def func(y, t, *p):
        return vf(y, t=t, p=p)

    ts = jnp.asarray([t0, t1])
    odeint_solution = jax.experimental.ode.odeint(
        func, u0, ts, *f_args, atol=1e-1 * atol, rtol=1e-1 * rtol
    )
    ys_reference = odeint_solution[-1, :]
    return t1, ys_reference


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases")
def fixture_solution_terminal_values(ode_problem, tolerances, solver):
    vf, u0, t0, t1, f_args = ode_problem
    atol, rtol = tolerances
    solution = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Checkpoint fixtures


@pytest_cases.fixture(scope="session", name="checkpoint_grid")
def fixture_checkpoint_grid(ode_problem):
    _, _, t0, t1, _ = ode_problem
    return jnp.linspace(t0, t1, endpoint=True, num=5)


@pytest_cases.fixture(scope="session", name="reference_checkpoints")
def fixture_reference_and_save_at(ode_problem, tolerances, checkpoint_grid):
    vf, (u0,), _, _, f_args = ode_problem
    atol, rtol = tolerances

    @jax.jit
    def func(y, t, *p):
        return vf(y, t=t, p=p)

    odeint_solution = jax.experimental.ode.odeint(
        func, u0, checkpoint_grid, *f_args, atol=1e-1 * atol, rtol=1e-1 * rtol
    )
    return checkpoint_grid, odeint_solution


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=_CHECKPT)
def fixture_solution_save_at(ode_problem, tolerances, solver, checkpoint_grid):
    vf, u0, _, _, f_args = ode_problem
    atol, rtol = tolerances
    solution = ivpsolve.solve_and_save_at(
        vf,
        u0,
        save_at=checkpoint_grid,
        parameters=f_args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve() fixtures


@pytest_cases.fixture(scope="session", name="solution_solve")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=_SOLVE)
def fixture_solution_solve(ode_problem, tolerances, solver):
    vf, u0, t0, t1, f_args = ode_problem
    atol, rtol = tolerances
    solution = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve_fixed_grid() fixtures


@pytest_cases.fixture(scope="session", name="fixed_grid")
def fixture_fixed_grid(ode_problem):
    _, _, t0, t1, _ = ode_problem
    return jnp.linspace(t0, t1, endpoint=True, num=10)


@pytest_cases.fixture(scope="session", name="solution_fixed_grid")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=_SOLVE)
def fixture_solution_fixed_grid(ode_problem, solver, fixed_grid):
    vf, u0, _, _, f_args = ode_problem
    solution = ivpsolve.solve_fixed_grid(
        vf,
        u0,
        grid=fixed_grid,
        parameters=f_args,
        solver=solver,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver
