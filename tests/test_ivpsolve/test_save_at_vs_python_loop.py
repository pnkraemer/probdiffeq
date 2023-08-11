"""Assert that solve_and_save_at is consistent with solve_with_python_loop()."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated, solution
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import filters
from tests.setup import setup


def test_save_at_result_matches_interpolated_adaptive_result():
    """Test that the save_at result matches the interpolation (using a filter)."""
    # Make a problem
    vf, u0, (t0, t1) = setup.ode()

    problem_args = (vf, u0)

    # Generate a solver
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    adaptive_kwargs = {
        "solver": solver,
        "dt0": 0.1,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_scale": jnp.ones_like(impl.ssm_util.prototype_output_scale()),
    }

    # Compute an adaptive solution and interpolate
    ts = jnp.linspace(t0, t1, num=15, endpoint=True)
    solution_adaptive = ivpsolve.solve_with_python_while_loop(
        *problem_args, t0=t0, t1=t1, **adaptive_kwargs
    )
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_adaptive, solver=solver
    )

    # Compute a save-at solution and remove the edge-points
    solution_save_at = ivpsolve.solve_and_save_at(
        *problem_args, save_at=ts, **adaptive_kwargs
    )

    u_save_at = solution_save_at.u[1:-1]
    marginals_save_at = jax.tree_util.tree_map(
        lambda s: s[1:-1], solution_save_at.marginals
    )

    # Assert similarity
    assert jnp.allclose(u_interp, u_save_at)
    assert testing.marginals_allclose(marginals_interp, marginals_save_at)
