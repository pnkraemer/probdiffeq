"""There are too many ways to smooth. We assert they all do the same."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes


@case
@parametrize("n", [2])
@parametrize("tol", [1e-2])
def smoother_fixpt_smoother_pair_eks0(n, tol):
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    fixpt_eks0 = recipes.dynamic_isotropic_fixpt_eks0(
        num_derivatives=n, atol=1e-2 * tol, rtol=tol
    )
    return eks0, fixpt_eks0


@pytest.mark.skip
@parametrize_with_cases("vf, u0, t0, t1, p", cases="..ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "eks, fixpt_eks", cases=".", prefix="smoother_fixpt_smoother_pair_"
)
def test_smoothing_checkpoint_equals_solver_state(vf, u0, t0, t1, p, eks, fixpt_eks):
    """In simulate_checkpoints(), if the checkpoint-grid equals the solution-grid\
     of a previous call to solve(), the results should be identical."""
    eks_sol = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=eks[0], info_op=eks[1]
    )

    # eks_sol.t is an adaptive grid
    # here, create an even grid which shares one point with the adaptive one.
    # This one point will be used for error-estimation.
    ts = _grid(eks_sol.t, t0=t0, t1=t1, factor=5)
    ts = ts[: ((len(ts) - 2) // 2)]
    print(ts)
    print()
    print()
    with jax.disable_jit():
        fixpt_eks_sol = ivpsolve.simulate_checkpoints(
            vf,
            u0,
            ts=ts,
            parameters=p,
            solver=fixpt_eks[0],
            info_op=fixpt_eks[1],
        )
    assert jnp.allclose(fixpt_eks_sol.t, ts)

    import matplotlib.pyplot as plt

    plt.plot(
        eks_sol.t,
        eks_sol.filtered.mean[:, -1, :],
        linestyle="None",
        marker="X",
        markersize=8,
        color="k",
        label="EKS",
    )
    plt.plot(
        fixpt_eks_sol.t,
        fixpt_eks_sol.filtered.mean[:, -1, :],
        # linestyle="None",
        linewidth=0.1,
        marker="P",
        markersize=12,
        alpha=0.5,
        label="FixPtEKS",
    )
    plt.legend()
    plt.ylim((-10, 10))
    plt.show()

    assert _tree_all_allclose(fixpt_eks_sol, eks_sol, atol=1e-2, rtol=1e-2)


# jit to reduce potential floating-point inconsistencies
@partial(jax.jit, static_argnames=["factor"])
def _grid(t_old, *, t0, t1, factor=1):
    midpoint = t_old[len(t_old) // 2]
    t_a = jnp.asarray([midpoint])
    t_b = jnp.linspace(t0, t1, num=factor * len(t_old), endpoint=True)
    ts = jnp.union1d(t_a, t_b, size=factor * len(t_old) + 1)
    return ts


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
