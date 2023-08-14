"""Test for mle calibration.

The posterior of the MLE solver is the same as for the calibration-free solver.
The output scale is different.
After applying solution.calibrate(), the posterior is different.
"""
import jax.numpy as jnp

from probdiffeq import ivpsolve, timestep
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated, solution, uncalibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="uncalibrated_solution")
def fixture_uncalibrated_solution():
    vf, u0, (t0, t1) = setup.ode()

    ibm = extrapolation.ibm_adaptive(num_derivatives=4)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "solver": solver,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }
    return ivpsolve.solve_and_save_every_step(*args, **kwargs)


@testing.fixture(name="mle_solution")
def fixture_mle_solution():
    vf, u0, (t0, t1) = setup.ode()

    ibm = extrapolation.ibm_adaptive(num_derivatives=4)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)

    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "solver": solver,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }
    return ivpsolve.solve_and_save_every_step(*args, **kwargs)


def test_python_loop_output_matches_diffrax(uncalibrated_solution, mle_solution):
    posterior_uncalibrated = uncalibrated_solution.posterior
    output_scale_uncalibrated = uncalibrated_solution.output_scale

    posterior_mle = mle_solution.posterior
    mle_output_scale = mle_solution.output_scale

    # Without a call to calibrate(), the posteriors are the same.
    assert testing.tree_all_allclose(posterior_uncalibrated, posterior_mle)
    assert not jnp.allclose(output_scale_uncalibrated, mle_output_scale)

    # With a call to calibrate(), the posteriors are different.
    posterior_calibrated = solution.calibrate(posterior_mle, mle_output_scale)
    assert not testing.tree_all_allclose(posterior_uncalibrated, posterior_calibrated)
