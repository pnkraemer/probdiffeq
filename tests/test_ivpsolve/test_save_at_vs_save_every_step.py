"""Assert that solve_adaptive_save_at is consistent with solve_with_python_loop()."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import functools, ode, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_save_at_result_matches_interpolated_adaptive_result(fact):
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy=strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = ivpsolvers.errorest_schober(
        prior=ibm, correction=ts0, atol=1e-2, rtol=1e-2, ssm=ssm
    )

    # Compute an adaptive solution and interpolate
    ts = np.linspace(t0, t1, num=15, endpoint=True)
    save_every = ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, errorest=errorest, solver=solver, dt0=0.1
    )
    offgrid = functools.vmap(lambda s: solver.offgrid_marginals(s, solution=save_every))
    u_interpolated = offgrid(ts[1:-1])

    # Compute a save-at solution and remove the edge-points
    save_at = ivpsolve.solve_adaptive_save_at(
        init, save_at=ts, errorest=errorest, solver=solver, dt0=0.1
    )
    u_save_at = tree_util.tree_map(lambda s: s[1:-1], save_at.u)

    # Assert similarity

    for ui, us in zip(u_interpolated.mean, u_save_at.mean):
        assert testing.allclose(ui, us)

    for ui, us in zip(u_interpolated.std, u_save_at.std):
        assert testing.allclose(ui, us)

    marginals_allclose_func = functools.partial(testing.marginals_allclose, ssm=ssm)
    marginals_allclose_func = functools.vmap(marginals_allclose_func)
    assert np.all(
        marginals_allclose_func(u_interpolated.marginals, u_save_at.marginals)
    )

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, u_save_at.mean)
    u_std_shape = tree_util.tree_map(np.shape, u_save_at.std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
