"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve() calls with default smoothers.
import jax
import jax.numpy as jnp
import pytest_cases

from odefilter import ivpsolve, solvers
from odefilter.implementations import recipes
from odefilter.strategies import filters, smoothers


@pytest_cases.case
def strategy_pair_smoother():
    impl = recipes.IsoTS0.from_params()
    return filters.Filter(impl), smoothers.Smoother(impl)


@pytest_cases.case
def strategy_pair_fixedpoint_smoother():
    impl = recipes.IsoTS0.from_params()
    return filters.Filter(impl), smoothers.FixedPointSmoother(impl)


@pytest_cases.parametrize_with_cases("fil, smo", cases=".", prefix="strategy_pair_")
def test_final_state_equal_to_filter(ode_problem, fil, smo):
    """Filters and smoothers should compute the same terminal values."""
    vf, u0, t0, t1, p = ode_problem

    atol, rtol = 1e-2, 1e-1
    filter_solution = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=fil),
        atol=atol,
        rtol=rtol,
    )
    smoother_solution = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=smo),
        atol=atol,
        rtol=rtol,
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    assert _tree_all_allclose(filter_solution.t, smoother_solution.t)
    assert _tree_all_allclose(filter_solution.u, smoother_solution.u)
    assert _tree_all_allclose(
        filter_solution.marginals.mean, smoother_solution.marginals.mean
    )
    assert _tree_all_allclose(
        cov(filter_solution.marginals.cov_sqrtm_lower),
        cov(smoother_solution.marginals.cov_sqrtm_lower),
    )
    assert _tree_all_allclose(
        filter_solution.output_scale_sqrtm, smoother_solution.output_scale_sqrtm
    )


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
