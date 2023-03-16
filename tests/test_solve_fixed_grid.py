"""Tests for solving IVPs on fixed grids."""
import functools

import jax
import jax.numpy as jnp
import pytest_cases

from probdiffeq import solution_routines, solvers, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


@pytest_cases.fixture(scope="session", name="solution_fixed_grid")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
@pytest_cases.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def fixture_solution_fixed_grid(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    t0, t1 = ode_problem.t0, ode_problem.t1
    grid = solver_config.grid_for_fixed_grid_fn(t0, t1)

    solution = solution_routines.solve_fixed_grid(
        ode_problem.vector_field,
        ode_problem.initial_values,
        grid=grid,
        parameters=ode_problem.args,
        solver=solver,
    )
    return (solution.t, solution.u), (grid, ode_problem.solution(grid))


def test_solve_fixed_grid_computes_terminal_values_correctly(
    solution_fixed_grid, solver_config
):
    (t, u), (t_ref, u_ref) = solution_fixed_grid

    assert jnp.allclose(
        t[-1],
        t_ref[-1],
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )
    assert jnp.allclose(
        u[-1],
        u_ref[-1],
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


# todo: all cases and all solvers
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize("impl_fn", [recipes.BlockDiagTS0.from_params])
@pytest_cases.parametrize("solver_fn", [solvers.MLESolver])
@pytest_cases.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def test_solve_fixed_grid_differentiable(
    ode_problem, solver_config, impl_fn, solver_fn, strat_fn
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=1,  # Low order traces more quickly
    )

    t0, t1 = ode_problem.t0, ode_problem.t1
    grid = solver_config.grid_for_fixed_grid_fn(t0, t1)

    fn = functools.partial(
        _parameter_to_solution,
        solver=solver,
        fixed_grid=grid,
        vf=ode_problem.vector_field,
        parameters=ode_problem.args,
    )

    # todo: use jax.test_util.check_jvp and *check_vjp
    # todo: split into two separate tests (jvp and vjp)
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
