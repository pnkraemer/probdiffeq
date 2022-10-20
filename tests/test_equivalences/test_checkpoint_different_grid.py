"""There are too many ways to smooth. We assert they all do the same."""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes


@case
@parametrize("n", [2])
def smoother_fixedpoint_smoother_pair_eks0(n):
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n)
    fixedpoint_eks0 = recipes.dynamic_isotropic_fixedpoint_eks0(num_derivatives=n)
    return eks0, fixedpoint_eks0


@parametrize_with_cases("vf, u0, t0, t1, p", cases="..ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "eks, fixedpoint_eks", cases=".", prefix="smoother_fixedpoint_smoother_pair_"
)
@parametrize("k", [1, 3])  # k * N // 2 off-grid points
def test_smoothing_checkpoint_equals_solver_state_smaller_grid(
    vf, u0, t0, t1, p, eks, fixedpoint_eks, k
):
    """In simulate_checkpoints(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve(), the results should be identical."""
    # eks_sol.t is an adaptive grid
    # here, create an even grid which shares one point with the adaptive one.
    # This one point will be used for error-estimation.

    args = (vf, u0)
    kwargs = {"parameters": p, "atol": 1e-1, "rtol": 1e-1}
    eks_sol = ivpsolve.solve(
        *args, t0=t0, t1=t1, solver=eks[0], info_op=eks[1], **kwargs
    )
    ts = jnp.linspace(t0, t1, num=k * len(eks_sol.t) // 2)
    dense = eks[0].offgrid_marginals_searchsorted(ts=ts[1:-1], solution=eks_sol)

    fp_eks_sol = ivpsolve.simulate_checkpoints(
        *args, ts=ts, solver=fixedpoint_eks[0], info_op=fixedpoint_eks[1], **kwargs
    )
    fixedpoint_eks_sol = fp_eks_sol[1:-1]  # reference is defined only on the interior

    # Compare all attributes for equality,
    # except for the covariance matrix square roots
    # which are equal modulo orthogonal transformation
    # (they are equal in square, though).
    # The backward models are not expected to be equal.
    assert jnp.allclose(fixedpoint_eks_sol.t, dense.t)
    assert jnp.allclose(fixedpoint_eks_sol.u, dense.u)
    assert jnp.allclose(fixedpoint_eks_sol.marginals.mean, dense.marginals.mean)
    assert jnp.allclose(fixedpoint_eks_sol.diffusion_sqrtm, dense.diffusion_sqrtm)

    # covariances are equal, but cov_sqrtm_lower might not be

    @jax.vmap
    def cov(x):
        return x @ x.T

    l0 = fixedpoint_eks_sol.marginals.cov_sqrtm_lower
    l1 = dense.marginals.cov_sqrtm_lower
    assert jnp.allclose(cov(l0), cov(l1))


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
