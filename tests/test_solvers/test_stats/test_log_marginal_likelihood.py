"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import ivpsolve
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing, tree_util
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solvers, stats
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="sol")
def fixture_sol():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = components.prior_ibm(num_derivatives=2)
    ts0 = components.correction_ts0()
    strategy = components.strategy_fixedpoint(ibm, ts0)
    solver = solvers.solver(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    save_at = np.linspace(t0, t1, endpoint=True, num=4)
    return ivpsolve.solve_adaptive_save_at(
        vf, init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1
    )


def test_output_is_a_scalar_and_not_nan_and_not_inf(sol):
    data = sol.u + 0.005
    lml = stats.log_marginal_likelihood(
        data, standard_deviation=np.ones_like(sol.t), posterior=sol.posterior
    )
    assert lml.shape == ()
    assert not np.isnan(lml)
    assert not np.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(sol):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving more standard-deviations than data-points.
    """
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=np.ones((k + 1,)), posterior=sol.posterior
        )


def test_that_function_raises_error_for_wrong_std_shape_wrong_ndim(sol):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving non-scalar standard-deviations.
    """
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=np.ones((k, 1)), posterior=sol.posterior
        )


def test_raises_error_for_terminal_values(sol):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    data = sol.u + 0.005

    posterior_t1 = tree_util.tree_map(lambda s: s[-1], sol)
    with testing.raises(ValueError, match="expected"):
        _ = stats.log_marginal_likelihood(
            data[-1], standard_deviation=np.ones_like(data[-1]), posterior=posterior_t1
        )


def test_raises_error_for_filter():
    """Non-terminal value calls are not possible for filters."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = components.prior_ibm(num_derivatives=2)
    ts0 = components.correction_ts0()
    strategy = components.strategy_filter(ibm, ts0)
    solver = solvers.solver(strategy)

    grid = np.linspace(t0, t1, num=3)
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)

    data = sol.u + 0.1
    std = np.ones((sol.u.shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = stats.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior
        )
