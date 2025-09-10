"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.fixture(name="solution")
@testing.parametrize("fact", ["isotropic", "blockdiag", "dense"])
def fixture_solution(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_fixedpoint(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
    save_at = np.linspace(t0, t1, endpoint=True, num=4)
    sol = ivpsolve.solve_adaptive_save_at(
        init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )
    return sol, ssm


def test_output_is_a_scalar_and_not_nan_and_not_inf(solution):
    sol, ssm = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u[0])
    std = tree_util.tree_map(np.ones_like, sol.u[0])
    lml = stats.log_marginal_likelihood(
        data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
    )
    assert lml.shape == ()
    assert not np.isnan(lml)
    assert not np.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(solution):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving more standard-deviations than data-points.
    """
    sol, ssm = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u[0])
    std = tree_util.tree_map(lambda s: np.ones_like(s[:-1]), sol.u[0])

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
        )


def test_that_function_raises_error_for_wrong_std_shape_wrong_ndim(solution):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving scalar standard-deviations.
    """
    sol, ssm = solution
    data = tree_util.tree_map(lambda s: s + 0.005, sol.u[0])
    std = tree_util.tree_map(lambda s: np.ones((len(s),)), sol.u[0])

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
        )


def test_raises_error_for_terminal_values(solution):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    sol, ssm = solution
    data = tree_util.tree_map(lambda s: s[-1] + 0.005, sol.u[0])
    std = tree_util.tree_map(lambda s: np.ones_like(s[-1]), sol.u[0])

    posterior_t1 = tree_util.tree_map(lambda s: s[-1], sol.posterior)
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=posterior_t1, ssm=ssm
        )


@testing.parametrize("fact", ["isotropic"])  # no dense/blockdiag because no impl test
def test_raises_error_for_filter(fact):
    """Non-terminal value calls are not possible for filters."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    grid = np.linspace(t0, t1, num=3)
    sol = ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)

    data = sol.u[0] + 0.1
    std = np.ones((sol.u[0].shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
        )
