"""Assert that solve_and_save_at is consistent with solve_with_python_loop()."""
import jax
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


def test_save_at_result_matches_interpolated_adaptive_result():
    """Test that the save_at result matches the interpolation (using a filter)."""
    # Make a problem
    vf, u0, (t0, t1) = setup.ode()

    # Generate a solver
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale=output_scale)
    problem_args = (vf, init)
    adaptive_kwargs = {"adaptive_solver": adaptive_solver, "dt0": 0.1}

    # Compute an adaptive solution and interpolate
    ts = jnp.linspace(t0, t1, num=15, endpoint=True)
    solution_adaptive = ivpsolve.solve_and_save_every_step(
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
    marginals_allclose_func = jax.vmap(testing.marginals_allclose)
    assert jnp.allclose(u_interp, u_save_at)
    assert jnp.all(marginals_allclose_func(marginals_interp, marginals_save_at))
