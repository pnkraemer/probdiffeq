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


@testing.case()
def case_solve_fixed_grid():
    vf, u0, (t0, t1) = setup.ode()
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "grid": jnp.linspace(t0, t1, endpoint=True, num=5),
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
    }

    def solver_to_solution(solver):
        return ivpsolve.solve_fixed_grid(*args, solver=solver, **kwargs)

    return solver_to_solution


@testing.case()
def case_solve_and_save_at():
    vf, u0, (t0, t1) = setup.ode()
    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "save_at": jnp.linspace(t0, t1, endpoint=True, num=5),
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }

    def solver_to_solution(solver):
        return ivpsolve.solve_and_save_at(*args, solver=solver, **kwargs)

    return solver_to_solution


@testing.case()
def case_solve_and_save_every_step():
    vf, u0, (t0, t1) = setup.ode()
    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }

    def solver_to_solution(solver):
        return ivpsolve.solve_and_save_every_step(*args, solver=solver, **kwargs)

    return solver_to_solution


@testing.case()
def case_simulate_terminal_values():
    vf, u0, (t0, t1) = setup.ode()
    dt0 = timestep.propose(lambda y: vf(y, t=t0), u0)
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=4)
    args = (vf, tcoeffs)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "output_scale": jnp.ones_like(impl.prototypes.output_scale()),
        "atol": 1e-2,
        "rtol": 1e-2,
        "dt0": dt0,
    }

    def solver_to_solution(solver):
        return ivpsolve.simulate_terminal_values(*args, solver=solver, **kwargs)

    return solver_to_solution


@testing.fixture(name="uncalibrated_and_mle_solution")
@testing.parametrize_with_cases("solver_to_solution", cases=".", prefix="case_")
def fixture_uncalibrated_and_mle_solution(solver_to_solution):
    ibm = extrapolation.ibm_adaptive(num_derivatives=4)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)

    uncalib = solver_to_solution(uncalibrated.solver(strategy))
    mle = solver_to_solution(calibrated.mle(strategy))
    return uncalib, mle


def test_python_loop_output_matches_diffrax(uncalibrated_and_mle_solution):
    uncalibrated_solution, mle_solution = uncalibrated_and_mle_solution

    posterior_uncalibrated = uncalibrated_solution.posterior
    output_scale_uncalibrated = uncalibrated_solution.output_scale

    posterior_mle = mle_solution.posterior
    output_scale_mle = mle_solution.output_scale

    # Without a call to calibrate(), the posteriors are the same.
    assert testing.tree_all_allclose(posterior_uncalibrated, posterior_mle)
    assert not jnp.allclose(output_scale_uncalibrated, output_scale_mle)

    # With a call to calibrate(), the posteriors are different.
    posterior_calibrated = solution.calibrate(posterior_mle, output_scale_mle)
    assert not testing.tree_all_allclose(posterior_uncalibrated, posterior_calibrated)
