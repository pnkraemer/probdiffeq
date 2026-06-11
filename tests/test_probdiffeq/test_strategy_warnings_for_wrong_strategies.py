"""Some strategies don't work with all solution routines."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import ode, testing
from probdiffeq.util import test_util


@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def test_warning_for_fixedpoint_in_save_every_step_mode(ssm_factory) -> None:
    vf, _, _ = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    ssm = ssm_factory()
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)

    with testing.warns():
        _ = test_util.solve_adaptive_save_every_step(error=error, solver=solver)


@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def test_warning_for_smoother_in_save_at_mode(ssm_factory) -> None:
    vf, _, _ = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    ssm = ssm_factory()
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedinterval()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)
    with testing.warns():
        _ = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
