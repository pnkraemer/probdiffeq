"""Tests for IVP solvers."""
import functools

import jax
import jax.numpy as jnp
import jax.tree_util
import pytest_cases

from odefilter import ivpsolve, solvers
from odefilter.implementations import isotropic
from odefilter.strategies import filters, smoothers


def test_solve_fixed_grid_computes_terminal_values_correctly(
    reference_terminal_values, solution_fixed_grid, tolerances
):
    t_ref, u_ref = reference_terminal_values
    atol, rtol = tolerances
    solution, _ = solution_fixed_grid

    assert jnp.allclose(solution.t[-1], t_ref)
    assert jnp.allclose(solution.u[-1], u_ref, atol=atol, rtol=rtol)


@pytest_cases.parametrize("strategy", [smoothers.Smoother, filters.Filter])
def test_solve_fixed_grid_differentiable(ode_problem, fixed_grid, strategy):
    vf, u0, t0, t1, f_args = ode_problem

    filter_or_smoother = strategy(
        # Low order because it traces/differentiates faster
        extrapolation=isotropic.IsoIBM.from_params(num_derivatives=1)
    )
    solver = solvers.Solver(strategy=filter_or_smoother, output_scale_sqrtm=1.0)

    fn = functools.partial(
        _parameter_to_solution,
        solver=solver,
        fixed_grid=fixed_grid,
        vf=vf,
        parameters=f_args,
    )

    fx = fn(u0[0])
    dfx_fwd = jax.jit(jax.jacfwd(fn, argnums=0))(u0[0])
    dfx_rev = jax.jit(jax.jacrev(fn, argnums=0))(u0[0])

    out_shape = _tree_shape(fx)
    in_shape = _tree_shape(u0[0])
    assert _tree_all_tree_map(jnp.allclose, dfx_fwd, dfx_rev)
    assert _tree_shape(dfx_fwd) == out_shape + in_shape


def _parameter_to_solution(u0, parameters, vf, solver, fixed_grid):
    solution = ivpsolve.solve_fixed_grid(
        vf, (u0,), ts=fixed_grid, parameters=parameters, solver=solver
    )
    return solution.u


def _tree_shape(tree):
    return jax.tree_util.tree_map(jnp.shape, tree)


def _tree_all_tree_map(fn, *tree):
    tree_of_bools = jax.tree_util.tree_map(fn, *tree)
    return jax.tree_util.tree_all(tree_of_bools)
