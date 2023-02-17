"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp
import pytest_cases
import pytest_cases.filters

from probdiffeq import dense_output
from probdiffeq.strategies import filters, smoothers


def test_offgrid_marginals_filter(solution_solve):
    solution, solver = solution_solve
    t0, t1 = solution.t[0], solution.t[-1]

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, filters.Filter):
        # Extrapolate from the left: close-to-left boundary must be similar,
        # but close-to-right boundary must not be similar
        u_left, _ = dense_output.offgrid_marginals(
            t=solution[0].t + 1e-4,
            solution=solution[1],
            solution_previous=solution[0],
            solver=solver,
        )
        u_right, _ = dense_output.offgrid_marginals(
            t=solution[1].t - 1e-4,
            solution=solution[1],
            solution_previous=solution[0],
            solver=solver,
        )
        assert jnp.allclose(u_left, solution[0].u, atol=1e-3, rtol=1e-3)
        assert not jnp.allclose(u_right, solution[0].u, atol=1e-3, rtol=1e-3)

        # Repeat the same but interpolating via *_searchsorted:
        # check we correctly landed in the first interval
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
        u, _ = dense_output.offgrid_marginals_searchsorted(
            ts=ts, solution=solution, solver=solver
        )
        assert jnp.allclose(u[0], solution.u[0], atol=1e-3, rtol=1e-3)
        assert not jnp.allclose(u[0], solution.u[1], atol=1e-3, rtol=1e-3)


def test_offgrid_marginals_smoother(solution_solve):
    solution, solver = solution_solve
    t0, t1 = solution.t[0], solution.t[-1]

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, smoothers.Smoother):
        # Extrapolate from the left: close-to-left boundary must be similar,
        # but close-to-right boundary must not be similar
        u_left, _ = dense_output.offgrid_marginals(
            t=solution[0].t + 1e-4,
            solution=solution[1],
            solution_previous=solution[0],
            solver=solver,
        )
        u_right, _ = dense_output.offgrid_marginals(
            t=solution[1].t - 1e-4,
            solution=solution[1],
            solution_previous=solution[0],
            solver=solver,
        )
        assert jnp.allclose(u_left, solution[0].u, atol=1e-3, rtol=1e-3)
        assert jnp.allclose(u_right, solution[1].u, atol=1e-3, rtol=1e-3)

        # Repeat the same but interpolating via *_searchsorted:
        # check we correctly landed in the first interval
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
        u, _ = dense_output.offgrid_marginals_searchsorted(
            ts=ts, solution=solution, solver=solver
        )
        assert jnp.allclose(u[0], solution.u[0], atol=1e-3, rtol=1e-3)
        assert jnp.allclose(u[-1], solution.u[-1], atol=1e-3, rtol=1e-3)


@pytest_cases.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_grid_samples(solution_save_at, shape):
    solution, solver = solution_save_at

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, smoothers.FixedPointSmoother):
        key = jax.random.PRNGKey(seed=15)
        u, samples = dense_output.sample(
            key, solution=solution, solver=solver, shape=shape
        )
        assert u.shape == shape + solution.u.shape
        assert samples.shape == shape + solution.marginals.hidden_state.sample_shape

        # Todo: test values of the samples by checking a chi2 statistic
        #  in terms of the joint posterior. But this requires a joint_posterior()
        #  method, which is only future work I guess. So far we use the eye-test
        #  in the notebooks, which looks good.


def test_negative_marginal_log_likelihood(solution_save_at):
    solution, solver = solution_save_at

    # todo: this is hacky. But the tests get faster?
    if isinstance(solver.strategy, smoothers.FixedPointSmoother):
        data = solution.u + 0.005
        k = solution.u.shape[0]

        mll = dense_output.negative_marginal_log_likelihood(
            observation_std=jnp.ones((k,)), u=data, solution=solution
        )
        assert mll.shape == ()
        assert not jnp.isnan(mll)
        assert not jnp.isinf(mll)
