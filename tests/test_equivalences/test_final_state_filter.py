"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve() calls with default smoothers.
import jax
import jax.numpy as jnp
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes, solvers
from odefilter.strategies import filters, smoothers


@case
def filter_smoother_pair_eks0():
    filter = filters.Filter()
    solver1 = solvers.DynamicSolver(filter)

    smoother = smoothers.Smoother()
    solver2 = solvers.DynamicSolver(smoother)
    return solver1, solver2


@case
def filter_smoother_pair_fixedpoint_eks0():
    filter = filters.Filter()
    solver1 = solvers.DynamicSolver(filter)

    smoother = smoothers.FixedPointSmoother()
    solver2 = solvers.DynamicSolver(smoother)
    return solver1, solver2


@parametrize_with_cases("vf, u0, t0, t1, p", cases="..ivp_cases", prefix="problem_")
@parametrize_with_cases("ekf, eks", cases=".", prefix="filter_smoother_pair_")
@parametrize("tol", [1e-1, 1e-3])
def test_final_state_equal_to_filter(vf, u0, t0, t1, p, ekf, eks, tol):
    """In simulate_terminal_values(), \
    every filter and smoother should yield the exact same result."""
    ekf_sol = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=ekf, atol=1e-2 * tol, rtol=tol
    )
    eks_sol = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=eks, atol=1e-2 * tol, rtol=tol
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
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
