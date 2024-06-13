"""Tests for IVP solvers."""

from probdiffeq import ivpsolve, solvers, stats
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.taylor import autodiff
from tests.setup import setup


def test_filter_marginals_close_only_to_left_boundary():
    """Assert that the filter-marginals interpolate well close to the left boundary."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = solvers.prior_ibm(num_derivatives=1)
    ts0 = solvers.correction_ts0()
    strategy = solvers.strategy_filter(ibm, ts0)
    solver = solvers.solver(strategy)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = (u0, vf(u0, t=t0))
    init = solver.initial_condition(tcoeffs, output_scale)
    grid = np.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = stats.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert np.allclose(u[0], sol.u[-2], atol=1e-3, rtol=1e-3)
    assert not np.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)


def test_smoother_marginals_close_to_both_boundaries():
    """Assert that the smoother-marginals interpolate well close to the boundary."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = solvers.prior_ibm(num_derivatives=4)
    ts0 = solvers.correction_ts0()
    strategy = solvers.strategy_smoother(ibm, ts0)
    solver = solvers.solver(strategy)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init = solver.initial_condition(tcoeffs, output_scale)
    grid = np.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)
    # Extrapolate from the left: close-to-left boundary must be similar,
    # and close-to-right boundary must be similar
    ts = np.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = stats.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)

    assert np.allclose(u[0], sol.u[-2], atol=1e-3, rtol=1e-3)
    assert np.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)
