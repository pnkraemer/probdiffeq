"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve() calls with default smoothers.
import jax
import jax.numpy as jnp
import pytest_cases

from odefilter import ivpsolve, solvers
from odefilter.strategies import filters, smoothers


@pytest_cases.case
def strategy_pair_smoother():
    return filters.Filter.from_params(), smoothers.Smoother.from_params()


@pytest_cases.case
def strategy_pair_fixedpoint_smoother():
    return filters.Filter.from_params(), smoothers.FixedPointSmoother.from_params()


@pytest_cases.parametrize_with_cases("ekf, eks", cases=".", prefix="strategy_pair_")
def test_final_state_equal_to_filter(ode_problem, ekf, eks):
    """Filters and smoothers should compute the same terminal values."""
    vf, u0, t0, t1, p = ode_problem

    atol, rtol = 1e-2, 1e-1
    ekf_sol = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=ekf),
        atol=atol,
        rtol=rtol,
    )
    eks_sol = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solvers.DynamicSolver(strategy=eks),
        atol=atol,
        rtol=rtol,
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    assert _tree_all_allclose(ekf_sol.t, eks_sol.t)
    assert _tree_all_allclose(ekf_sol.u, eks_sol.u)
    assert _tree_all_allclose(ekf_sol.marginals.mean, eks_sol.marginals.mean)
    assert _tree_all_allclose(
        cov(ekf_sol.marginals.cov_sqrtm_lower), cov(eks_sol.marginals.cov_sqrtm_lower)
    )
    assert _tree_all_allclose(ekf_sol.output_scale_sqrtm, eks_sol.output_scale_sqrtm)


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
