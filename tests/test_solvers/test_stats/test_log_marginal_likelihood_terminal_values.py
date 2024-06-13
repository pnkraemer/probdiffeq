"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solvers, stats, strategies
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.case()
def case_strategy_filter():
    return strategies.filter_adaptive


@testing.case()
def case_strategy_smoother():
    return strategies.smoother_adaptive


@testing.case()
def case_strategy_fixedpoint():
    return strategies.fixedpoint_adaptive


@testing.fixture(name="sol")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
def fixture_sol(strategy_func):
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = components.prior_ibm(num_derivatives=4)
    ts0 = components.correction_ts0()
    strategy = strategy_func(ibm, ts0)
    solver = solvers.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.simulate_terminal_values(
        vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1
    )


def test_output_is_scalar_and_not_inf_and_not_nan(sol):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    data = sol.u + 0.1
    mll = stats.log_marginal_likelihood_terminal_values(
        data, standard_deviation=np.asarray(1e-2), posterior=sol.posterior
    )
    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_terminal_values_error_for_wrong_shapes(sol):
    data = sol.u + 0.005

    # Non-scalar observation std
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood_terminal_values(
            data, standard_deviation=np.ones((1,)), posterior=sol.posterior
        )

    # Data does not match u
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood_terminal_values(
            data[None, ...], standard_deviation=np.ones(()), posterior=sol.posterior
        )
