"""The fixedpoint-smoother and smoother should yield identical results.

That is, when called with correct adaptive- and checkpoint-setups.
"""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import smoothers
from tests.setup import setup


@testing.fixture(name="solver_setup")
def fixture_solver_setup():
    vf, (u0,), (t0, t1) = setup.ode()

    output_scale = jnp.ones_like(impl.ssm_util.prototype_output_scale())
    args = (vf, (u0,))
    kwargs = {"atol": 1e-3, "rtol": 1e-3, "output_scale": output_scale, "dt0": 0.1}
    return args, kwargs, (t0, t1)


@testing.fixture(name="solution_smoother")
def fixture_solution_smoother(solver_setup):
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    args, kwargs, (t0, t1) = solver_setup
    return ivpsolve.solve_and_save_every_step(
        *args, t0=t0, t1=t1, solver=solver, **kwargs
    )


def test_fixedpoint_smoother_equivalent_same_grid(solver_setup, solution_smoother):
    """Test that with save_at=smoother_solution.t, the results should be identical."""
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = smoothers.fixedpoint_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    save_at = solution_smoother.t
    args, kwargs, _ = solver_setup
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=save_at, solver=solver, **kwargs
    )
    assert testing.tree_all_allclose(solution_fixedpoint, solution_smoother)


def test_fixedpoint_smoother_equivalent_different_grid(solver_setup, solution_smoother):
    """Test that the interpolated smoother result equals the save_at result."""
    args, kwargs, _ = solver_setup
    save_at = solution_smoother.t

    # Re-generate the smoothing solver
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = smoothers.smoother_adaptive(ibm, ts0)
    solver_smoother = uncalibrated.solver(strategy)

    # Compute the offgrid-marginals
    ts = jnp.linspace(save_at[0], save_at[-1], num=7, endpoint=True)
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_smoother, solver=solver_smoother
    )

    # Generate a fixedpoint solver and solve (saving at the interpolation points)
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = smoothers.fixedpoint_adaptive(ibm, ts0)
    solver_fixedpoint = uncalibrated.solver(strategy)
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=ts, solver=solver_fixedpoint, **kwargs
    )

    # Extract the interior points of the save_at solution
    # (because only there is the interpolated solution defined)
    u_fixedpoint = solution_fixedpoint.u[1:-1]
    marginals_fixedpoint = jax.tree_util.tree_map(
        lambda s: s[1:-1], solution_fixedpoint.marginals
    )

    # Compare QOI and marginals
    assert testing.tree_all_allclose(u_fixedpoint, u_interp)
    assert testing.marginals_allclose(marginals_fixedpoint, marginals_interp)
