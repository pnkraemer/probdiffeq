"""There are too many ways to smooth. We assert they all do the same."""

import jax.numpy as jnp
import pytest
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes

# todo: reuse solve() calls with default smoothers.


@case(tags=())
@parametrize("n", [2])
@parametrize("tol", [1e-1])
def smoother_fixpt_smoother_pair_eks0(n, tol):
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    fixpt_eks0 = recipes.dynamic_isotropic_fixpt_eks0(
        num_derivatives=n, atol=1e-2 * tol, rtol=tol
    )
    return eks0, fixpt_eks0


@case(tags=["works"])
@parametrize("n", [2])
@parametrize("tol", [1e-1])
def smoother_fixpt_smoother_pair_two_eks0(n, tol):
    # if the checkpoints are equal to the solver states,
    # then the checkpoint-simulator replicates _exactly_ what the non-checkpoint-
    # smoother does. So the tests must also pass in this setup.
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    return eks0, eks0


@pytest.mark.skip
@parametrize_with_cases("vf, u0, t0, t1, p", cases="..ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "eks, fixpt_eks",
    cases=".",
    prefix="smoother_fixpt_smoother_pair_",
    has_tag=["works"],
)
def test_smoothing_checkpoint_equals_solver_state(vf, u0, t0, t1, p, eks, fixpt_eks):
    """In simulate_checkpoints(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve(), the results should be identical."""
    eks_sol = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=eks[0], info_op=eks[1]
    )
    fixpt_eks_sol = ivpsolve.simulate_checkpoints(
        vf,
        u0,
        ts=eks_sol.t,
        parameters=p,
        solver=fixpt_eks[0],
        info_op=fixpt_eks[1],
    )

    import matplotlib.pyplot as plt

    plt.plot(
        eks_sol.t,
        eks_sol.filtered.mean[:, -1, :],
        # linestyle="None",
        marker="o",
        markersize=6,
        label="EKS",
    )
    plt.plot(
        fixpt_eks_sol.t,
        fixpt_eks_sol.filtered.mean[:, -1, :],
        # linestyle="None",
        marker="^",
        markersize=6,
        label="FixPtEKS",
    )
    plt.legend()
    plt.ylim((-10, 10))
    plt.show()
    assert False
    print(eks_sol.filtered.mean - fixpt_eks_sol.filtered.mean)
    print()

    is_allclose = _tree_allclose(fixpt_eks_sol, eks_sol)
    print(is_allclose.t)
    print(is_allclose.u)
    print(is_allclose.filtered.mean)
    print(is_allclose.filtered.cov_sqrtm_lower)
    print(is_allclose.backward_model)

    assert _tree_all_allclose(fixpt_eks_sol, eks_sol, atol=1e-1, rtol=1e-1)


@pytest.mark.skip
def test_smoothing_coarser_checkpoints():
    """Similar to the above, but the checkpoint grid is coarser than the solver-grid."""


@pytest.mark.skip
def test_smoothing_finer_checkpoints():
    """Similar to the above, but the checkpoint grid is finer than the solver-grid."""


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
