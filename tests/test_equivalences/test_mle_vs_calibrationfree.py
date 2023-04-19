"""Tests for solver equivalences."""

import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize("strategy_fn", [filters.Filter, smoothers.FixedPointSmoother])
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
    ts = jnp.linspace(ode_problem.t0, ode_problem.t1, num=11)  # magic number
    kwargs = {"save_at": ts, "parameters": ode_problem.args, "atol": 1e-1, "rtol": 1e-1}

    mle_sol = ivpsolve.solve_and_save_at(
        *args, solver=mlesolver, output_scale=1.0, **kwargs
    )

    free_sol = ivpsolve.solve_and_save_at(
        *args, solver=freesolver, output_scale=mle_sol.output_scale, **kwargs
    )

    assert jnp.allclose(mle_sol.t, free_sol.t)
    assert jnp.allclose(mle_sol.u, free_sol.u)
    assert jnp.allclose(mle_sol.marginals.mean, free_sol.marginals.mean)
    assert jnp.allclose(
        mle_sol.marginals.cov_sqrtm_lower, free_sol.marginals.cov_sqrtm_lower
    )
    # todo: assert backward models?
