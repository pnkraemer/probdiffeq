"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.case()
def case_strategy_filter():
    return probdiffeq.strategy_filter


@testing.case()
def case_strategy_smoother():
    return probdiffeq.strategy_smoother


@testing.case()
def case_strategy_fixedpoint():
    return probdiffeq.strategy_fixedpoint


@testing.fixture(name="solution")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution(strategy_func, fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.correction_ts0(vf, ssm=ssm)
    strategy = strategy_func(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)
    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, errorest=errorest)
    sol = solve(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)
    return sol, strategy


def test_output_is_scalar_and_not_inf_and_not_nan(solution):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    sol, strategy = solution

    data = tree_util.tree_map(lambda s: s + 0.005, sol.u.mean[0])
    std = tree_util.tree_map(lambda _s: 1e-2 * np.ones(()), sol.u.std[0])

    mll = strategy.log_marginal_likelihood_terminal_values(
        data, standard_deviation=std, posterior=sol.posterior
    )

    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_raise_error_if_structures_dont_match(solution):
    sol, strategy = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u.mean[0])
    std = 1.0  # not the correct pytree

    with testing.raises(ValueError, match="structure"):
        _ = strategy.log_marginal_likelihood_terminal_values(
            data, standard_deviation=std, posterior=sol.posterior
        )
