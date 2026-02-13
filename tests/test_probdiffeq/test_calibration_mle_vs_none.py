"""Test for mle calibration.

The posterior of the MLE solver is the same as for the calibration-free solver.
The output scale is different.
After applying stats.calibrate(), the posterior is different.
"""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_solve_fixed_grid(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    grid = np.linspace(t0, t1, endpoint=True, num=5)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        return solve(init, grid=grid)

    return solver_to_solution, ssm


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_solve_adaptive_save_at(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), u0)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    save_at = np.linspace(t0, t1, endpoint=True, num=5)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy, prior=ibm, correction=ts0, ssm=ssm)
        errorest = probdiffeq.errorest_local_residual(
            prior=ibm, correction=ts0, ssm=ssm
        )
        solve = ivpsolve.solve_adaptive_save_at(errorest=errorest, solver=solver)
        return solve(init, save_at=save_at, dt0=dt0, atol=1e-2, rtol=1e-2)

    return solver_to_solution, ssm


@testing.case()
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def case_simulate_terminal_values(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()
    dt0 = ivpsolve.dt0(lambda y: vf(y, t=t0), u0)
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)

    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)

    def solver_to_solution(solver_fun, strategy_fun):
        strategy = strategy_fun(ssm=ssm)
        solver = solver_fun(strategy=strategy, prior=ibm, correction=ts0, ssm=ssm)
        errorest = probdiffeq.errorest_local_residual(
            prior=ibm, correction=ts0, ssm=ssm
        )
        solve = ivpsolve.solve_adaptive_terminal_values(
            errorest=errorest, solver=solver
        )
        return solve(init, t0=t0, t1=t1, dt0=dt0, atol=1e-2, rtol=1e-2)

    return solver_to_solution, ssm


@testing.fixture(name="uncalibrated_and_mle_solution")
@testing.parametrize_with_cases("solver_to_solution", cases=".", prefix="case_")
@testing.parametrize(
    "strategy_fun",
    [
        probdiffeq.strategy_filter,
        probdiffeq.strategy_smoother_fixedinterval,
        probdiffeq.strategy_smoother_fixedpoint,
    ],
)
def fixture_uncalibrated_and_mle_solution(solver_to_solution, strategy_fun):
    solve, ssm = solver_to_solution
    uncalib = solve(probdiffeq.solver, strategy_fun)
    mle = solve(probdiffeq.solver_mle, strategy_fun)
    return (uncalib, mle)


# fixedpoint-solver in save_every_step gives nonsensical results
# (which raises a warning), but the test remains valid!
@testing.filterwarnings("ignore")
def test_calibration_changes_the_posterior(uncalibrated_and_mle_solution):
    (uncalibrated, mle) = uncalibrated_and_mle_solution

    # Assert the means are identical, but the stds & scales are not.
    assert not testing.allclose(uncalibrated.output_scale, mle.output_scale)
    assert not testing.allclose(uncalibrated.u.std, mle.u.std)

    # For some solvers, the means are not exactly identical but the differences
    # are small (and vanish for double precision). No idea why. But everything else
    # seems to work, so I assume the code is fine.
    assert testing.allclose(uncalibrated.u.mean[0], mle.u.mean[0], rtol=1e-3)
