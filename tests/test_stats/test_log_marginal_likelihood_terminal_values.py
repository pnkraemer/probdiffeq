"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


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
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = strategy_func(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
    sol = ivpsolve.solve_adaptive_terminal_values(
        init, t0=t0, t1=t1, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )
    return sol, ssm


def test_output_is_scalar_and_not_inf_and_not_nan(solution):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    sol, ssm = solution

    data = tree_util.tree_map(lambda s: s + 0.005, sol.u[0])
    std = tree_util.tree_map(lambda _s: 1e-2 * np.ones(()), sol.u[0])

    mll = stats.log_marginal_likelihood_terminal_values(
        data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
    )

    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_raise_error_if_structures_dont_match(solution):
    sol, ssm = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u[0])
    std = 1.0  # not the correct pytree

    with testing.raises(ValueError, match="structure"):
        _ = stats.log_marginal_likelihood_terminal_values(
            data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
        )
