"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp
import pytest
import pytest_cases
import pytest_cases.filters

from probdiffeq import dense_output, solution_routines, test_util
from probdiffeq.strategies import filters, smoothers


@pytest_cases.fixture(scope="session", name="solution_native_python_while_loop")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_native_python_while_loop(ode_problem):
    solver = test_util.generate_solver(num_derivatives=1)
    solution = solution_routines.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return solution, solver


def test_solution_is_iterable(solution_native_python_while_loop):
    solution, _ = solution_native_python_while_loop
    assert isinstance(solution[0], type(solution))
    assert len(solution) == len(solution.t)


def test_getitem_raises_error_for_nonbatched_solutions(
    solution_native_python_while_loop,
):
    """__getitem__ only works for batched solutions."""
    solution, _ = solution_native_python_while_loop
    with pytest.raises(ValueError):
        _ = solution[0][0]
    with pytest.raises(ValueError):
        _ = solution[0, 0]


def test_loop_over_solution_is_possible(solution_native_python_while_loop):
    solution, _ = solution_native_python_while_loop

    i = 0
    for i, sol in zip(range(2 * len(solution)), solution):
        assert isinstance(sol, type(solution))

    assert i == len(solution) - 1


# Maybe this test should be in a different test suite, but it does not really matter...
def test_marginal_nth_derivative_of_solution(solution_native_python_while_loop):
    solution, _ = solution_native_python_while_loop

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = solution.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with pytest.raises(ValueError):
        solution.marginals.marginal_nth_derivative(100)


def test_offgrid_marginals_filter(solution_native_python_while_loop):
    solution, solver = solution_native_python_while_loop
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


def test_offgrid_marginals_smoother(solution_native_python_while_loop):
    solution, solver = solution_native_python_while_loop
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


@pytest_cases.fixture(scope="session", name="solution_save_at")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_solution_save_at(ode_problem):
    solver = test_util.generate_solver(strategy_factory=smoothers.FixedPointSmoother)

    save_at = jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=4)
    solution = solution_routines.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=save_at,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return solution, solver


@pytest_cases.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_grid_samples(solution_save_at, shape):
    solution, solver = solution_save_at

    key = jax.random.PRNGKey(seed=15)
    u, samples = dense_output.sample(key, solution=solution, solver=solver, shape=shape)
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
