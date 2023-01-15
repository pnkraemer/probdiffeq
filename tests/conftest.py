"""Test configurations."""

import jax
import jax.experimental.ode
import jax.numpy as jnp
import pytest_cases
import pytest_cases.filters

from probdiffeq import ivpsolve, taylor

# Set some test filters


def is_filter(cf):
    (case_tags,) = pytest_cases.filters.get_case_tags(cf)
    return case_tags.strategy == "filter"


def is_smoother(cf):
    (case_tags,) = pytest_cases.filters.get_case_tags(cf)
    return case_tags.strategy == "smoother"


def is_fixedpoint(cf):
    (case_tags,) = pytest_cases.filters.get_case_tags(cf)
    return case_tags.strategy == "fixedpoint"


def can_simulate_terminal_values(cf):
    return is_filter(cf) | is_fixedpoint(cf) | is_smoother(cf)


def can_solve_and_save_at(cf):
    return is_filter(cf) | is_fixedpoint(cf)


def can_solve(cf):
    return is_filter(cf) | is_smoother(cf)


# Common fixtures (for example, tolerances)


@pytest_cases.fixture(scope="session", name="tolerances")
def fixture_tolerances():
    return 1e-5, 1e-3


# Terminal value fixtures


@pytest_cases.fixture(scope="session", name="reference_terminal_values")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_reference_terminal_values(ode_problem):
    return ode_problem.t1, ode_problem.solution(ode_problem.t1)


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases(
    "solver", cases=".solver_cases", filter=can_simulate_terminal_values
)
def fixture_solution_terminal_values(ode_problem, tolerances, solver):
    atol, rtol = tolerances
    solution = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Checkpoint fixtures


@pytest_cases.fixture(scope="session", name="checkpoint_grid")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_checkpoint_grid(ode_problem):
    return jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=5)


@pytest_cases.fixture(scope="session", name="reference_checkpoints")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_reference_and_save_at(ode_problem, checkpoint_grid):
    return checkpoint_grid, jax.vmap(ode_problem.solution)(checkpoint_grid)


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases(
    "solver", cases=".solver_cases", filter=can_solve_and_save_at
)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_save_at(ode_problem, tolerances, solver, checkpoint_grid):
    atol, rtol = tolerances
    solution = ivpsolve.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=checkpoint_grid,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve() fixtures


@pytest_cases.fixture(scope="session", name="solution_solve")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=can_solve)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_solve(ode_problem, tolerances, solver):
    atol, rtol = tolerances
    solution = ivpsolve.solve(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1 * atol,
        rtol=1e-1 * rtol,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve_fixed_grid() fixtures


@pytest_cases.fixture(scope="session", name="fixed_grid")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_fixed_grid(ode_problem):
    return jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=10)


@pytest_cases.fixture(scope="session", name="solution_fixed_grid")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=can_solve)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_fixed_grid(ode_problem, solver, fixed_grid):
    solution = ivpsolve.solve_fixed_grid(
        ode_problem.vector_field,
        ode_problem.initial_values,
        grid=fixed_grid,
        parameters=ode_problem.args,
        solver=solver,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver
