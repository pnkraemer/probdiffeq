"""Some strategies don't work with all solution routines."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import ode, testing
from probdiffeq.util import test_util


@testing.parametrize("fact", ["isotropic"])  # no dense/blockdiag because no impl test
def test_warning_for_fixedpoint_in_save_every_step_mode(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = probdiffeq.correction_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)

    with testing.warns():
        _ = test_util.solve_adaptive_save_every_step(errorest=errorest, solver=solver)


@testing.parametrize("fact", ["isotropic"])  # no dense/blockdiag because no impl test
def test_warning_for_smoother_in_save_at_mode(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.correction_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)
    with testing.warns():
        _ = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
