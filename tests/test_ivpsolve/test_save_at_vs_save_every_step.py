"""Assert that solve_adaptive_save_at is consistent with solve_with_python_loop()."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import functools, ode, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_save_at_result_matches_interpolated_adaptive_result(fact):
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    problem_args = (vf, init)
    adaptive_kwargs = {"adaptive_solver": adaptive_solver, "dt0": 0.1, "ssm": ssm}

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

    u_save_at = tree_util.tree_map(lambda s: s[1:-1], solution_save_at.u)
    marginals_save_at = tree_util.tree_map(
        lambda s: s[1:-1], solution_save_at.marginals
    )

    # Assert similarity

    for ui, us in zip(u_interp, u_save_at):
        assert np.allclose(ui, us)

    marginals_allclose_func = functools.partial(testing.marginals_allclose, ssm=ssm)
    marginals_allclose_func = functools.vmap(marginals_allclose_func)
    assert np.all(marginals_allclose_func(marginals_interp, marginals_save_at))

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, solution_save_at.u)
    u_std_shape = tree_util.tree_map(np.shape, solution_save_at.u_std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
