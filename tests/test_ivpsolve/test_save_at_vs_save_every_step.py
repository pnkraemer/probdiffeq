"""Assert that solve_adaptive_save_at is consistent with solve_with_python_loop()."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import functools, testing, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


def test_save_at_result_matches_interpolated_adaptive_result(ssm):
    """Test that the save_at result matches the interpolation (using a filter)."""
    # Make a problem
    vf, u0, (t0, t1) = ssm.default_ode

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale=output_scale)

    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    init = solver.initial_condition()
    problem_args = (vf, init)
    adaptive_kwargs = {"adaptive_solver": adaptive_solver, "dt0": 0.1}

    # Compute an adaptive solution and interpolate
    ts = np.linspace(t0, t1, num=15, endpoint=True)
    solution_adaptive = ivpsolve.solve_adaptive_save_every_step(
        *problem_args, t0=t0, t1=t1, **adaptive_kwargs
    )
    u_interp, marginals_interp = stats.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_adaptive, solver=solver
    )

    # Compute a save-at solution and remove the edge-points
    solution_save_at = ivpsolve.solve_adaptive_save_at(
        *problem_args, save_at=ts, **adaptive_kwargs
    )

    u_save_at = solution_save_at.u[1:-1]
    marginals_save_at = tree_util.tree_map(
        lambda s: s[1:-1], solution_save_at.marginals
    )

    # Assert similarity
    marginals_allclose_func = functools.vmap(testing.marginals_allclose)
    assert np.allclose(u_interp, u_save_at)
    assert np.all(marginals_allclose_func(marginals_interp, marginals_save_at))
