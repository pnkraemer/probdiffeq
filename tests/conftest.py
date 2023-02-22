"""Test configurations."""

import dataclasses

import jax
import jax.experimental.ode
import jax.numpy as jnp
import pytest_cases
import pytest_cases.filters

from probdiffeq import solution_routines, taylor

# Set some test filters

# todo: remove those.


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


# Solver configurations (for example, tolerances.)
# My attempt at bundling up all those magic save_at grids, tolerances, etc.


@dataclasses.dataclass
class SolverConfiguration:
    atol_solve: float
    rtol_solve: float
    grid_for_fixed_grid: jax.Array
    grid_for_save_at: jax.Array

    @property
    def atol_assert(self):
        return 10 * self.atol_solve

    @property
    def rtol_assert(self):
        return 10 * self.rtol_solve


@pytest_cases.fixture(scope="session", name="solver_config")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solver_config(ode_problem):
    grid = jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=10)
    save_at = jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=5)
    return SolverConfiguration(
        atol_solve=1e-5,
        rtol_solve=1e-3,
        grid_for_fixed_grid=grid,
        grid_for_save_at=save_at,
    )


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
def fixture_solution_terminal_values(ode_problem, solver_config, solver):
    solution = solution_routines.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Checkpoint fixtures


@pytest_cases.fixture(scope="session", name="reference_checkpoints")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_reference_save_at(ode_problem, solver_config):
    xs = solver_config.grid_for_save_at
    return xs, jax.vmap(ode_problem.solution)(xs)


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases(
    "solver", cases=".solver_cases", filter=can_solve_and_save_at
)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_save_at(ode_problem, solver_config, solver):
    solution = solution_routines.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=solver_config.grid_for_save_at,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve() fixtures


@pytest_cases.fixture(scope="session", name="solution_solve")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=can_solve)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_solve_with_python_while_loop(ode_problem, solver_config, solver):
    solution = solution_routines.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


# Solve_fixed_grid() fixtures


@pytest_cases.fixture(scope="session", name="solution_fixed_grid")
@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases", filter=can_solve)
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_fixed_grid(ode_problem, solver, solver_config):
    solution = solution_routines.solve_fixed_grid(
        ode_problem.vector_field,
        ode_problem.initial_values,
        grid=solver_config.grid_for_fixed_grid,
        parameters=ode_problem.args,
        solver=solver,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver
