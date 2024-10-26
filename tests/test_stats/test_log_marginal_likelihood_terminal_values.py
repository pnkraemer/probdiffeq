"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing


@testing.case()
def case_strategy_filter():
    return ivpsolvers.strategy_filter


@testing.case()
def case_strategy_smoother():
    return ivpsolvers.strategy_smoother


@testing.case()
def case_strategy_fixedpoint():
    return ivpsolvers.strategy_fixedpoint


@testing.fixture(name="solution")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution(strategy_func, fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = strategy_func(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver(strategy, ssm=ssm)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    init = solver.initial_condition()
    sol = ivpsolve.solve_adaptive_terminal_values(
        vf, init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )
    return sol, ssm


def test_output_is_scalar_and_not_inf_and_not_nan(solution):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    sol, ssm = solution
    data = sol.u[0] + 0.1
    mll = stats.log_marginal_likelihood_terminal_values(
        data, standard_deviation=np.asarray(1e-2), posterior=sol.posterior, ssm=ssm
    )
    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_terminal_values_error_for_wrong_shapes(solution):
    sol, ssm = solution
    data = sol.u[0] + 0.005

    # Non-scalar observation std
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood_terminal_values(
            data, standard_deviation=np.ones((1,)), posterior=sol.posterior, ssm=ssm
        )

    # Data does not match u
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood_terminal_values(
            data[None, ...],
            standard_deviation=np.ones(()),
            posterior=sol.posterior,
            ssm=ssm,
        )
