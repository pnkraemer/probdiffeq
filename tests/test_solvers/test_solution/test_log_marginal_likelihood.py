"""Tests for log-marginal-likelihood functionality."""
import jax
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import filters, fixedpoint
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="sol")
def fixture_sol():
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = fixedpoint.fixedpoint_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init = solver.initial_condition(tcoeffs, output_scale)

    save_at = jnp.linspace(t0, t1, endpoint=True, num=4)
    return ivpsolve.solve_and_save_at(
        vf, init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1
    )


def test_output_is_a_scalar_and_not_nan_and_not_inf(sol):
    data = sol.u + 0.005
    lml = solution.log_marginal_likelihood(
        data, standard_deviation=jnp.ones_like(sol.t), posterior=sol.posterior
    )
    assert lml.shape == ()
    assert not jnp.isnan(lml)
    assert not jnp.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(sol):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving more standard-deviations than data-points.
    """
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = solution.log_marginal_likelihood(
            data, standard_deviation=jnp.ones((k + 1,)), posterior=sol.posterior
        )


def test_that_function_raises_error_for_wrong_std_shape_wrong_ndim(sol):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving non-scalar standard-deviations.
    """
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = solution.log_marginal_likelihood(
            data, standard_deviation=jnp.ones((k, 1)), posterior=sol.posterior
        )


def test_raises_error_for_terminal_values(sol):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    data = sol.u + 0.005

    posterior_t1 = jax.tree_util.tree_map(lambda s: s[-1], sol)
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood(
            data[-1], standard_deviation=jnp.ones_like(data[-1]), posterior=posterior_t1
        )


def test_raises_error_for_filter():
    """Non-terminal value calls are not possible for filters."""
    vf, (u0,), (t0, t1) = setup.ode()

    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = uncalibrated.solver(strategy)

    grid = jnp.linspace(t0, t1, num=3)
    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)

    data = sol.u + 0.1
    std = jnp.ones((sol.u.shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = solution.log_marginal_likelihood(
            data, standard_deviation=std, posterior=sol.posterior
        )
