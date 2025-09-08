"""Test for mle calibration.

The posterior of the MLE solver is the same as for the calibration-free solver.
The output scale is different.
After applying stats.calibrate(), the posterior is different.
"""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_solve_fixed_grid(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    grid = np.linspace(t0, t1, endpoint=True, num=5)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        return ivpsolve.solve_fixed_grid(init, solver=solver, grid=grid, ssm=ssm)

    return solver_to_solution, ssm


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_solve_adaptive_save_at(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), u0)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    save_at = np.linspace(t0, t1, endpoint=True, num=5)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
        return ivpsolve.solve_adaptive_save_at(
            init, adaptive_solver=adaptive_solver, save_at=save_at, dt0=dt0, ssm=ssm
        )

    return solver_to_solution, ssm


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_solve_adaptive_save_every_step(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), u0)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
        return ivpsolve.solve_adaptive_save_every_step(
            init, adaptive_solver=adaptive_solver, t0=t0, t1=t1, dt0=dt0, ssm=ssm
        )

    return solver_to_solution, ssm


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_simulate_terminal_values(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()
    dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), u0)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm, atol=1e-2, rtol=1e-2)
        return ivpsolve.solve_adaptive_terminal_values(
            init, adaptive_solver=adaptive_solver, t0=t0, t1=t1, dt0=dt0, ssm=ssm
        )

    return solver_to_solution, ssm


@testing.fixture(name="uncalibrated_and_mle_solution")
@testing.parametrize_with_cases("solver_to_solution", cases=".", prefix="case_")
@testing.parametrize(
    "strategy_fun", [ivpsolvers.strategy_filter, ivpsolvers.strategy_fixedpoint]
)
def fixture_uncalibrated_and_mle_solution(solver_to_solution, strategy_fun):
    solve, ssm = solver_to_solution
    uncalib = solve(ivpsolvers.solver, strategy_fun)
    mle = solve(ivpsolvers.solver_mle, strategy_fun)
    return (uncalib, mle), ssm


# fixedpoint-solver in save_every_step gives nonsensical results
# (which raises a warning), but the test remains valid!
@testing.filterwarnings("ignore")
def test_calibration_changes_the_posterior(uncalibrated_and_mle_solution):
    (uncalibrated_solution, mle_solution), ssm = uncalibrated_and_mle_solution

    posterior_uncalibrated = uncalibrated_solution.posterior
    output_scale_uncalibrated = uncalibrated_solution.output_scale

    posterior_mle = mle_solution.posterior
    output_scale_mle = mle_solution.output_scale

    # Without a call to calibrate(), the posteriors are the same.
    assert testing.tree_all_allclose(posterior_uncalibrated, posterior_mle)
    assert not np.allclose(output_scale_uncalibrated, output_scale_mle)

    # With a call to calibrate(), the posteriors are different.
    posterior_calibrated = stats.calibrate(posterior_mle, output_scale_mle, ssm=ssm)
    assert not testing.tree_all_allclose(posterior_uncalibrated, posterior_calibrated)
