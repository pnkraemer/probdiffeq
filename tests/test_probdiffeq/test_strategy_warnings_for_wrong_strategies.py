"""Some strategies don't work with all solution routines."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import ode, testing
from probdiffeq.util import test_util


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_warning_for_fixedpoint_in_save_every_step_mode(fact) -> None:
    vf, (u0,), (t0, _t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    _init, iwp = ssm.prior_wiener_integrated(tcoeffs)

    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)

    with testing.warns():
        _ = test_util.solve_adaptive_save_every_step(error=error, solver=solver)


@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_warning_for_smoother_in_save_at_mode(fact) -> None:
    vf, (u0,), (t0, _t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, [u0], t=t0)
    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    _init, iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedinterval()
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)
    with testing.warns():
        _ = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
