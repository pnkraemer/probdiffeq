"""Tests for marginal log likelihood functionality (terminal values)."""
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import filters, fixedpoint, smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.case()
def case_strategy_filter():
    return filters.filter_adaptive


@testing.case()
def case_strategy_smoother():
    return smoothers.smoother_adaptive


@testing.case()
def case_strategy_fixedpoint():
    return fixedpoint.fixedpoint_adaptive


@testing.fixture(name="sol")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
def fixture_sol(strategy_func):
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=4)
    ts0 = corrections.ts0()
    strategy = strategy_func(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    return ivpsolve.simulate_terminal_values(
        vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1
    )


def test_output_is_scalar_and_not_inf_and_not_nan(sol):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    data = sol.u + 0.1
    mll = solution.log_marginal_likelihood_terminal_values(
        data, standard_deviation=jnp.asarray(1e-2), posterior=sol.posterior
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


def test_terminal_values_error_for_wrong_shapes(sol):
    data = sol.u + 0.005

    # Non-scalar observation std
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            data, standard_deviation=jnp.ones((1,)), posterior=sol.posterior
        )

    # Data does not match u
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            data[None, ...], standard_deviation=jnp.ones(()), posterior=sol.posterior
        )
