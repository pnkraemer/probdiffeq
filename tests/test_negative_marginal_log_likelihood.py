"""Tests for IVP solvers."""
import jax.numpy as jnp
import pytest
import pytest_cases
import pytest_cases.filters

from probdiffeq import dense_output, solution_routines, solvers, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


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


def test_negative_marginal_log_likelihood(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005
    k = solution.u.shape[0]

    mll = dense_output.negative_marginal_log_likelihood(
        observation_std=jnp.ones((k,)), u=data, solution=solution
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


def test_negative_marginal_log_likelihood_error_for_wrong_std_shape_1(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005
    k = solution.u.shape[0]

    with pytest.raises(ValueError, match="does not match"):
        _ = dense_output.negative_marginal_log_likelihood(
            observation_std=jnp.ones((k + 1,)), u=data, solution=solution
        )


def test_negative_marginal_log_likelihood_error_for_wrong_std_shape_2(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005
    k = solution.u.shape[0]

    with pytest.raises(ValueError, match="does not match"):
        _ = dense_output.negative_marginal_log_likelihood(
            observation_std=jnp.ones((k, 1)), u=data, solution=solution
        )


def test_negative_marginal_log_likelihood_error_for_wrong_u_shape_1(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005
    k = solution.u.shape[0]

    with pytest.raises(ValueError, match="does not match"):
        _ = dense_output.negative_marginal_log_likelihood(
            observation_std=jnp.ones((k,)), u=data[..., None], solution=solution
        )


def test_negative_marginal_log_likelihood_error_for_terminal_values(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005

    with pytest.raises(ValueError, match="expected"):
        _ = dense_output.negative_marginal_log_likelihood(
            observation_std=jnp.ones_like(data[-1]), u=data[-1], solution=solution[-1]
        )


def test_negative_marginal_log_likelihood_terminal_values(solution_save_at):
    solution, solver = solution_save_at
    data = solution.u + 0.005

    mll = dense_output.negative_marginal_log_likelihood_terminal_values(
        observation_std=jnp.ones(()), u=data[-1], solution=solution[-1]
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


def test_negative_marginal_log_likelihood_terminal_values_error_for_wrong_shapes(
    solution_save_at,
):
    solution, solver = solution_save_at
    data = solution.u + 0.005

    with pytest.raises(ValueError, match="expected"):
        _ = dense_output.negative_marginal_log_likelihood_terminal_values(
            observation_std=jnp.ones((1,)), u=data[-1], solution=solution[-1]
        )

    with pytest.raises(ValueError, match="does not match"):
        _ = dense_output.negative_marginal_log_likelihood_terminal_values(
            observation_std=jnp.ones(()), u=data[-1, None], solution=solution[-1]
        )

    with pytest.raises(ValueError, match="expected"):
        _ = dense_output.negative_marginal_log_likelihood_terminal_values(
            observation_std=jnp.ones(()), u=data[-1:], solution=solution[-1:]
        )


@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize("strategy_fn", [filters.Filter, smoothers.Smoother])
def test_filter_ts0_iso_terminal_value_nll(ode_problem, strategy_fn):
    """Issue #477."""
    recipe = recipes.IsoTS0.from_params(num_derivatives=4)
    strategy = strategy_fn(recipe)
    solver = solvers.CalibrationFreeSolver(strategy, output_scale_sqrtm=1.0)
    sol = solution_routines.simulate_terminal_values(
        ode_problem.vector_field,
        initial_values=ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
    )
    data = sol.u + 0.1
    mll = dense_output.negative_marginal_log_likelihood_terminal_values(
        observation_std=1e-2, u=data, solution=sol
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)
