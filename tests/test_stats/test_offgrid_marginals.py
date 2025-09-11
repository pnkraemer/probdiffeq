"""Tests for IVP solvers."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_filter_marginals_close_only_to_left_boundary(fact):
    """Assert that the filter-marginals interpolate well close to the left boundary."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = (u0, vf(u0, t=t0))
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    grid = np.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = stats.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    for u1, u2 in zip(u, sol.u):
        u1_ = tree_util.tree_map(lambda s: s[0], u1)
        u2_ = tree_util.tree_map(lambda s: s[-2], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)

        u1_ = tree_util.tree_map(lambda s: s[-1], u1)
        u2_ = tree_util.tree_map(lambda s: s[-1], u2)
        assert not testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_smoother_marginals_close_to_both_boundaries(fact):
    """Assert that the smoother-marginals interpolate well close to the boundary."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_smoother(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    grid = np.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)
    # Extrapolate from the left: close-to-left boundary must be similar,
    # and close-to-right boundary must be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = stats.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)

    for u1, u2 in zip(u, sol.u):
        u1_ = tree_util.tree_map(lambda s: s[0], u1)
        u2_ = tree_util.tree_map(lambda s: s[-2], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)

        u1_ = tree_util.tree_map(lambda s: s[-1], u1)
        u2_ = tree_util.tree_map(lambda s: s[-1], u2)
        assert testing.allclose(u1_, u2_, atol=1e-3, rtol=1e-3)
