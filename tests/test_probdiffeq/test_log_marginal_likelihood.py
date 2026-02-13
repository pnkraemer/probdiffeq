"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.fixture(name="solution")
@testing.parametrize("fact", ["isotropic", "blockdiag", "dense"])
def fixture_solution(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_local_residual(prior=ibm, correction=ts0, ssm=ssm)
    save_at = np.linspace(t0, t1, endpoint=True, num=4)
    solve = ivpsolve.solve_adaptive_save_at(errorest=errorest, solver=solver)
    sol = solve(init, save_at=save_at, atol=1e-2, rtol=1e-2)
    return sol, strategy


def test_output_is_a_scalar_and_not_nan_and_not_inf(solution):
    sol, strategy = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u.mean[0])
    std = tree_util.tree_map(lambda _s: np.ones_like(sol.t), sol.u.std[0])
    lml = strategy.log_marginal_likelihood(
        data, standard_deviation=std, posterior=sol.posterior
    )
    assert lml.shape == ()
    assert not np.isnan(lml)
    assert not np.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(solution):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving fewer standard-deviations than data-points.
    """
    sol, strategy = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u.mean[0])
    std = tree_util.tree_map(lambda _s: np.ones_like(sol.t[:-1]), sol.u.std[0])

    with testing.raises(ValueError, match="does not match"):
        _ = strategy.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior
        )


def test_raises_error_for_terminal_values(solution):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    sol, strategy = solution
    data = tree_util.tree_map(lambda s: s[-1] + 0.005, sol.u.mean[0])
    std = tree_util.tree_map(lambda _s: np.ones_like(sol.t[-1]), sol.u.std[0])

    posterior_t1 = tree_util.tree_map(lambda s: s[-1], sol.posterior)
    with testing.raises(ValueError, match="expected"):
        _ = strategy.log_marginal_likelihood(
            data, standard_deviation=std, posterior=posterior_t1
        )


@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def test_raises_error_for_filter(fact):
    """Non-terminal value calls are not possible for filters."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    grid = np.linspace(t0, t1, num=3)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    sol = solve(init, grid=grid)
    data = tree_util.tree_map(lambda s: s + 0.1, sol.u.mean[0])
    std = tree_util.tree_map(np.ones_like, sol.u.std[0])
    with testing.raises(TypeError, match="ilter"):
        _ = strategy.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior
        )


def test_raise_error_if_structures_dont_match(solution):
    sol, strategy = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u.mean[0])
    std = np.ones_like(sol.t)  # not the correct pytree

    with testing.raises(ValueError, match="tree structure"):
        _ = strategy.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior
        )
