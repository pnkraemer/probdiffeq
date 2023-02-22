"""Tests for solving IVPs for checkpoints."""

import jax.numpy as jnp
import pytest
import pytest_cases

from probdiffeq import solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers


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
