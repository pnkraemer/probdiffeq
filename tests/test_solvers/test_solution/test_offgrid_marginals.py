"""Tests for IVP solvers."""
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import filters, smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


def test_filter_marginals_close_only_to_left_boundary():
    """Assert that the filter-marginals interpolate well close to the left boundary."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=1)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = (u0, vf(u0, t=t0))
    init = solver.initial_condition(tcoeffs, output_scale)
    grid = jnp.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = jnp.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert jnp.allclose(u[0], sol.u[-2], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)


def test_smoother_marginals_close_to_both_boundaries():
    """Assert that the smoother-marginals interpolate well close to the boundary."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=4)
    ts0 = corrections.ts0()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=4)
    init = solver.initial_condition(tcoeffs, output_scale)
    grid = jnp.linspace(t0, t1, endpoint=True, num=5)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)
    # Extrapolate from the left: close-to-left boundary must be similar,
    # and close-to-right boundary must be similar
    ts = jnp.linspace(sol.t[-2] + 1e-4, sol.t[-1] - 1e-4, num=5, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)

    assert jnp.allclose(u[0], sol.u[-2], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)
