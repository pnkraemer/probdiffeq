"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.fixture(name="solution")
@testing.parametrize("fact", ["isotropic", "blockdiag", "dense"])
def fixture_solution(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver(strategy, ssm=ssm)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    init = solver.initial_condition()

    save_at = np.linspace(t0, t1, endpoint=True, num=4)
    sol = ivpsolve.solve_adaptive_save_at(
        vf, init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
    )
    return sol, ssm


def test_output_is_a_scalar_and_not_nan_and_not_inf(solution):
    sol, ssm = solution
    data = sol.u[0] + 0.005
    lml = stats.log_marginal_likelihood(
        data, standard_deviation=np.ones_like(sol.t), posterior=sol.posterior, ssm=ssm
    )
    assert lml.shape == ()
    assert not np.isnan(lml)
    assert not np.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(solution):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving more standard-deviations than data-points.
    """
    sol, ssm = solution
    data = sol.u[0] + 0.005
    k = sol.u[0].shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=np.ones((k + 1,)), posterior=sol.posterior, ssm=ssm
        )


def test_that_function_raises_error_for_wrong_std_shape_wrong_ndim(solution):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving non-scalar standard-deviations.
    """
    sol, ssm = solution
    data = sol.u[0] + 0.005
    k = sol.u[0].shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=np.ones((k, 1)), posterior=sol.posterior, ssm=ssm
        )


def test_raises_error_for_terminal_values(solution):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    sol, ssm = solution
    data = sol.u[0] + 0.005

    posterior_t1 = tree_util.tree_map(lambda s: s[-1], sol)
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood(
            data[-1],
            standard_deviation=np.ones_like(data[-1]),
            posterior=posterior_t1,
            ssm=ssm,
        )


@testing.parametrize("fact", ["isotropic"])  # no dense/blockdiag because no impl test
def test_raises_error_for_filter(fact):
    """Non-terminal value calls are not possible for filters."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver(strategy, ssm=ssm)

    grid = np.linspace(t0, t1, num=3)
    init = solver.initial_condition()
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver, ssm=ssm)

    data = sol.u[0] + 0.1
    std = np.ones((sol.u[0].shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior, ssm=ssm
        )
