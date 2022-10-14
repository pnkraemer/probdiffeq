"""There are too many ways to smooth. We assert they all do the same."""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_all, tree_map
from pytest_cases import case, parametrize, parametrize_with_cases

from odefilter import ivpsolve, recipes


@case
@parametrize("n", [1, 3])
@parametrize("tol", [1e-1, 1e-3])
def filter_smoother_pair_eks0(n, tol):
    ekf0 = recipes.dynamic_isotropic_ekf0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    return ekf0, eks0


@case
@parametrize("n", [1, 3])
@parametrize("tol", [1e-1, 1e-3])
def filter_smoother_pair_fixpt_eks0(n, tol):
    ekf0 = recipes.dynamic_isotropic_ekf0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    eks0 = recipes.dynamic_isotropic_fixpt_eks0(
        num_derivatives=n, atol=1e-2 * tol, rtol=tol
    )
    return ekf0, eks0


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases("ekf, eks", cases=".", prefix="filter_smoother_pair_")
def test_final_state_equal_to_filter(vf, u0, t0, t1, p, ekf, eks):
    """In simulate_terminal_values(), \
    every filter and smoother should yield the exact same result."""
    ekf_sol = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=ekf[0], info_op=ekf[1]
    )
    eks_sol = ivpsolve.simulate_terminal_values(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=eks[0], info_op=eks[1]
    )

    assert _tree_all_allclose(ekf_sol.t, eks_sol.t)
    assert _tree_all_allclose(ekf_sol.u, eks_sol.u)
    assert _tree_all_allclose(ekf_sol.filtered, eks_sol.filtered)
    assert _tree_all_allclose(ekf_sol.diffusion_sqrtm, eks_sol.diffusion_sqrtm)


@case
@parametrize("n", [1, 3])
@parametrize("tol", [1e-1, 1e-3])
def smoother_fixpt_smoother_pair_eks0(n, tol):
    eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=n, atol=1e-2 * tol, rtol=tol)
    fixpt_eks0 = recipes.dynamic_isotropic_fixpt_eks0(
        num_derivatives=n, atol=1e-2 * tol, rtol=tol
    )
    return eks0, fixpt_eks0


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "eks, fixpt_eks", cases=".", prefix="smoother_fixpt_smoother_pair_"
)
def test_smoothing_checkpoint_equals_solver_state(vf, u0, t0, t1, p, eks, fixpt_eks):
    """In simulate_checkpoints(), if the checkpoint-grid equals the solution-grid
    of a previous call to solve(), the results should be identical."""

    eks_sol = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=eks[0], info_op=eks[1]
    )
    with jax.disable_jit():
        fixpt_eks_sol = ivpsolve.simulate_checkpoints(
            vf,
            u0,
            ts=eks_sol.t,
            parameters=p,
            solver=fixpt_eks[0],
            info_op=fixpt_eks[1],
        )
    allclose = _tree_allclose(fixpt_eks_sol, eks_sol)

    print(allclose.t)
    print(allclose.u)
    print(allclose.filtered.mean)
    print(allclose.filtered.cov_sqrtm_lower)
    print(allclose.backward_model.transition)
    print(allclose.backward_model.noise.mean)
    print(allclose.backward_model.noise.cov_sqrtm_lower)

    assert _tree_all_allclose(fixpt_eks_sol, eks_sol)


def test_smoothing_coarser_checkpoints():
    pass


def test_smoothing_finer_checkpoints():
    pass


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return tree_map(allclose_partial, tree1, tree2)
