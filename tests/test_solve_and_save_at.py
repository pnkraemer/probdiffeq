"""Tests for solving IVPs for checkpoints."""

import jax.numpy as jnp
import pytest
import pytest_cases

from probdiffeq import solution_routines, solvers, taylor, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
@pytest_cases.parametrize("strat_fn", [filters.Filter, smoothers.FixedPointSmoother])
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_save_at(ode_problem, solver_fn, impl_fn, strat_fn, solver_config):
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=(2,),
        num_derivatives=4,
    )

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


def test_save_at_solved_correctly(
    reference_checkpoints, solution_save_at, solver_config
):
    t_ref, u_ref = reference_checkpoints
    solution, _ = solution_save_at

    assert jnp.allclose(solution.t, t_ref)
    assert jnp.allclose(
        solution.u,
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def test_smoother_warning(ode_problem):
    """A non-fixed-point smoother is not usable in save-at-simulation."""
    ts = jnp.linspace(ode_problem.t0, ode_problem.t1, num=3)
    solver = solvers.DynamicSolver(smoothers.Smoother(recipes.IsoTS0.from_params()))

    # todo: does this compute the full solve? We only want to catch a warning!
    with pytest.warns():
        solution_routines.solve_and_save_at(
            ode_problem.vector_field,
            ode_problem.initial_values,
            save_at=ts,
            parameters=ode_problem.args,
            solver=solver,
        )
