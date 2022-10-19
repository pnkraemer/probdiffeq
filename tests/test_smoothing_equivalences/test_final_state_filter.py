"""There are too many ways to smooth. We assert they all do the same."""

# todo: reuse solve() calls with default smoothers.
import jax
import jax.numpy as jnp
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes


@case
@parametrize("n", [2, 3])
def filter_smoother_pair_eks0(n):
    ekf0 = recipes.dynamic_isotropic_ekf0(num_derivatives=n)
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n)
    return ekf0, eks0


@case
@parametrize("n", [2, 3])
def filter_smoother_pair_fixedpt_eks0(n):
    ekf0 = recipes.dynamic_isotropic_ekf0(num_derivatives=n)
    eks0 = recipes.dynamic_isotropic_fixedpt_eks0(num_derivatives=n)
    return ekf0, eks0


@parametrize_with_cases("vf, u0, t0, t1, p", cases="..ivp_cases", prefix="problem_")
@parametrize_with_cases("ekf, eks", cases=".", prefix="filter_smoother_pair_")
@parametrize("tol", [1e-1, 1e-3])
def test_final_state_equal_to_filter(vf, u0, t0, t1, p, ekf, eks, tol):
    """In simulate_terminal_values(), \
    every filter and smoother should yield the exact same result."""
    ekf_sol = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=ekf[0],
        info_op=ekf[1],
        atol=1e-2 * tol,
        rtol=tol,
    )
    eks_sol = ivpsolve.simulate_terminal_values(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=eks[0],
        info_op=eks[1],
        atol=1e-2 * tol,
        rtol=tol,
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
    assert _tree_all_allclose(ekf_sol.diffusion_sqrtm, eks_sol.diffusion_sqrtm)


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
