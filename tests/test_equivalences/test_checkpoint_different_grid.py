"""There are too many ways to smooth. We assert they all do the same."""

import jax
import jax.numpy as jnp
import pytest_cases

from probdiffeq import dense_output, solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers

# todo: both this file and test_checkpoint_same_grid.py call
#  solve_with_python_while_loop(... solver=smo) and solve_and_save_at(solver=fp_smo)
#  this redundancy should be eliminated


@pytest_cases.case
def smoother_pair_smoother_and_fixedpoint():
    impl = recipes.IsoTS0.from_params()
    return smoothers.Smoother(impl), smoothers.FixedPointSmoother(impl)


@pytest_cases.parametrize_with_cases("smo, fp_smo", cases=".", prefix="smoother_pair_")
@pytest_cases.parametrize("k", [1, 3])  # k * N // 2 off-grid points
@pytest_cases.parametrize_with_cases("ode_problem", cases="..problem_cases")
def test_smoothing_checkpoint_equals_solver_state(ode_problem, smo, fp_smo, k):
    """In solve_and_save_at(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve_with_python_while_loop(), \
     the results should be identical."""
    # smo_sol.t is an adaptive grid
    # here, create an even grid which shares one point with the adaptive one.
    # This one point will be used for error-estimation.

    args = (ode_problem.vector_field, ode_problem.initial_values)
    kwargs = {"parameters": ode_problem.args, "atol": 1e-1, "rtol": 1e-1}
    smo_sol = solution_routines.solve_with_python_while_loop(
        *args,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        solver=solvers.DynamicSolver(strategy=smo),
        **kwargs
    )
    ts = jnp.linspace(ode_problem.t0, ode_problem.t1, num=k * len(smo_sol.t) // 2)
    u, dense = dense_output.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=smo_sol, solver=solvers.DynamicSolver(strategy=smo)
    )

    fp_smo_sol = solution_routines.solve_and_save_at(
        *args, save_at=ts, solver=solvers.DynamicSolver(strategy=fp_smo), **kwargs
    )
    fixedpoint_smo_sol = fp_smo_sol[1:-1]  # reference is defined only on the interior

    # Compare all attributes for equality,
    # except for the covariance matrix square roots
    # which are equal modulo orthogonal transformation
    # (they are equal in square, though).
    # The backward models are not expected to be equal.
    assert jnp.allclose(fixedpoint_smo_sol.t, ts[1:-1])
    assert jnp.allclose(fixedpoint_smo_sol.u, u)
    assert jnp.allclose(
        fixedpoint_smo_sol.marginals.hidden_state.mean, dense.hidden_state.mean
    )

    # covariances are equal, but cov_sqrtm_lower might not be

    @jax.vmap
    def cov(x):
        return x @ x.T

    l0 = fixedpoint_smo_sol.marginals.hidden_state.cov_sqrtm_lower
    l1 = dense.hidden_state.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1))


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
