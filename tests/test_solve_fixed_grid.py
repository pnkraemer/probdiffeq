"""Tests for solving IVPs on fixed grids."""
import functools

import jax
import jax.numpy as jnp
import pytest_cases

from probdiffeq import solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


def test_solve_fixed_grid_computes_terminal_values_correctly(
    reference_terminal_values, solution_fixed_grid, solver_config
):
    t_ref, u_ref = reference_terminal_values
    solution, _ = solution_fixed_grid

    assert jnp.allclose(solution.t[-1], t_ref)
    assert jnp.allclose(
        solution.u[-1],
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


@pytest_cases.parametrize("strategy", [smoothers.Smoother, filters.Filter])
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def test_solve_fixed_grid_differentiable(ode_problem, solver_config, strategy):
    # Low order because it traces & differentiates faster
    filter_or_smoother = strategy(
        implementation=recipes.IsoTS0.from_params(num_derivatives=1)
    )
    solver = solvers.CalibrationFreeSolver(
        strategy=filter_or_smoother, output_scale_sqrtm=1.0
    )

    fn = functools.partial(
        _parameter_to_solution,
        solver=solver,
        fixed_grid=solver_config.grid_for_fixed_grid,
        vf=ode_problem.vector_field,
        parameters=ode_problem.args,
    )

    fx = fn(ode_problem.initial_values[0])
    dfx_fwd = jax.jit(jax.jacfwd(fn, argnums=0))(ode_problem.initial_values[0])
    dfx_rev = jax.jit(jax.jacrev(fn, argnums=0))(ode_problem.initial_values[0])

    out_shape = _tree_shape(fx)
    in_shape = _tree_shape(ode_problem.initial_values[0])
    assert _tree_all_tree_map(jnp.allclose, dfx_fwd, dfx_rev)
    assert _tree_shape(dfx_fwd) == out_shape + in_shape


def _parameter_to_solution(u0, parameters, vf, solver, fixed_grid):
    solution = solution_routines.solve_fixed_grid(
        vf, (u0,), grid=fixed_grid, parameters=parameters, solver=solver
    )
    return solution.u


def _tree_shape(tree):
    return jax.tree_util.tree_map(jnp.shape, tree)


def _tree_all_tree_map(fn, *tree):
    tree_of_bools = jax.tree_util.tree_map(fn, *tree)
    return jax.tree_util.tree_all(tree_of_bools)
