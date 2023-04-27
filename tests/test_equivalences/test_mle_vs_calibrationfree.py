"""Tests for solver equivalences."""

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother_fixedpoint])
def test_mle_vs_calibrationfree(ode_problem, strategy_fn):
    """Assert equivalence between MLESolver and CalibrationFreeSolver.

    The MLE-solution must be identical to the calibration-free solution
    provided the latter is started with the MLE of the former.

    This must hold for all strategies and all state-space factorisations.
    """
    impl = recipes.ts0_iso()
    strategy = strategy_fn(*impl)

    mlesolver = ivpsolvers.MLESolver(strategy)
    freesolver = ivpsolvers.CalibrationFreeSolver(strategy)

    args = (ode_problem.vector_field, ode_problem.initial_values)
    ts = jnp.linspace(
        ode_problem.t0, ode_problem.t1, endpoint=True, num=5
    )  # magic number
    kwargs = {"save_at": ts, "parameters": ode_problem.args, "atol": 1e-1, "rtol": 1e-1}

    # Solve-and-saveat contains the most init/extract (i.e. rescaling) logic,
    # so this is what we use for the test.
    solution_mle = ivpsolve.solve_and_save_at(
        *args, solver=mlesolver, output_scale=1.0, **kwargs
    )
    solution_free = ivpsolve.solve_and_save_at(
        *args, solver=freesolver, output_scale=solution_mle.output_scale[-1], **kwargs
    )

    # The following is essentially an assert _tree_allclose(solution1, solution2),
    # but the backward model choleskies are only equal in square
    # (they have different diagonal signs), so we have to compare them more manually.

    assert _tree_all_allclose(solution_mle.t, solution_free.t)
    assert _tree_all_allclose(solution_mle.u, solution_free.u)
    assert _tree_all_allclose(solution_mle.output_scale, solution_free.output_scale)
    assert _tree_all_allclose(solution_mle.marginals, solution_free.marginals)
    assert _tree_all_allclose(solution_mle.num_steps, solution_free.num_steps)

    # If we are smoothing, we also compare the backward models.
    if isinstance(strategy, smoothers._FixedPointSmoother):  # noqa: E731
        rand_mle = solution_mle.posterior
        rand_free = solution_free.posterior
        assert _tree_all_allclose(rand_mle.init, rand_free.init)

        bw_mle = rand_mle.backward_model
        bw_free = rand_free.backward_model
        assert _tree_all_allclose(bw_mle.transition, bw_free.transition)
        assert _tree_all_allclose(bw_mle.noise.mean, bw_free.noise.mean)

        @jax.vmap
        def square(x):
            return jnp.dot(x, x.T)

        cholesky_mle = bw_mle.noise.cov_sqrtm_lower
        cholesky_free = bw_free.noise.cov_sqrtm_lower

        assert jnp.allclose(square(cholesky_mle), square(cholesky_free))


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
