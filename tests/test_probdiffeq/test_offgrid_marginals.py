"""Tests for IVP solvers."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, testing, tree
from probdiffeq.util import test_util


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_save_at_result_matches_interpolated_adaptive_result(fact) -> None:
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_iwp(ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)

    # Compute an adaptive solution and interpolate
    ts = np.linspace(t0, t1, num=15, endpoint=True)
    solve = test_util.solve_adaptive_save_every_step(error=error, solver=solver)
    save_every = solve(init, t0=t0, t1=t1, dt0=0.1, atol=1e-2, rtol=1e-2)
    offgrid = func.vmap(lambda s: solver.offgrid_marginals(s, solution=save_every))
    u_interpolated = func.jit(offgrid)(ts[1:-1])

    # Compute a save-at solution and remove the edge-points
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    save_at = func.jit(solve)(init, atol=1e-2, rtol=1e-2, save_at=ts, dt0=0.1)
    u_save_at = tree.tree_map(lambda s: s[1:-1], save_at.u)

    # Assert similarity

    for ui, us in zip(u_interpolated.mean, u_save_at.mean):
        assert testing.allclose(ui, us)

    for ui, us in zip(u_interpolated.std, u_save_at.std):
        assert testing.allclose(ui, us)

    marginals_allclose_func = func.vmap(testing.marginals_allclose)
    are_close = marginals_allclose_func(u_interpolated.marginals, u_save_at.marginals)

    assert np.all(are_close)

    # Assert u and u_std have matching shapes (that was wrong before)
    _, u_shape = tree.tree_flatten(u_save_at.mean)
    _, u_std_shape = tree.tree_flatten(u_save_at.std)
    match = tree.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree.tree_all(match)


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_filter_marginals_close_only_to_left_boundary(fact) -> None:
    """Assert that the filter-marginals interpolate well close to the left boundary."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = (u0, vf(u0, t=t0))
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_iwp(ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    grid = np.linspace(t0, t1, endpoint=True, num=5)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    sol = solve(init, grid=grid)

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    offgrid_marginals = func.partial(solver.offgrid_marginals, solution=sol)
    u = func.vmap(offgrid_marginals)(ts)
    for u1, u2 in zip(u.mean, sol.u.mean):
        u1_ = tree.tree_map(lambda s: s[0], u1)
        u2_ = tree.tree_map(lambda s: s[-2], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)

        u1_ = tree.tree_map(lambda s: s[-1], u1)
        u2_ = tree.tree_map(lambda s: s[-1], u2)
        assert not testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_smoother_marginals_close_to_both_boundaries(fact) -> None:
    """Assert that the smoother-marginals interpolate well close to the boundary."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_iwp(ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)

    grid = np.linspace(t0, t1, endpoint=True, num=5)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    sol = solve(init, grid=grid)

    # Extrapolate from the left: close-to-left boundary must be similar,
    # and close-to-right boundary must be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    offgrid_marginals = func.partial(solver.offgrid_marginals, solution=sol)
    u = func.vmap(offgrid_marginals)(ts)

    for u1, u2 in zip(u.mean, sol.u.mean):
        u1_ = tree.tree_map(lambda s: s[0], u1)
        u2_ = tree.tree_map(lambda s: s[-2], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)

        u1_ = tree.tree_map(lambda s: s[-1], u1)
        u2_ = tree.tree_map(lambda s: s[-1], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)
