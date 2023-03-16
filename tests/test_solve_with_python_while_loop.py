"""Tests for solving IVPs on adaptive grids."""
import jax.numpy as jnp
import pytest
import pytest_cases

from probdiffeq import solution_routines, taylor, test_util
from probdiffeq.strategies import filters, smoothers


@pytest_cases.fixture(scope="session", name="solution_solve")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
@pytest_cases.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def fixture_solution_solve_with_python_while_loop(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=(2,),
        num_derivatives=4,
    )
    solution = solution_routines.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
    )
    return solution, solver


def test_solve_computes_correct_terminal_value(
    reference_terminal_values, solution_solve, solver_config
):
    t_ref, u_ref = reference_terminal_values
    solution, _ = solution_solve

    assert jnp.allclose(solution.t[-1], t_ref)
    assert jnp.allclose(
        solution.u[-1],
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


def test_solution_is_iterable(solution_solve):
    solution, _ = solution_solve

    assert isinstance(solution[0], type(solution))
    assert len(solution) == len(solution.t)


def test_getitem_raises_error_for_nonbatched_solutions(solution_solve):
    solution, _ = solution_solve

    # __getitem__ only works for batched solutions.
    with pytest.raises(ValueError):
        _ = solution[0][0]
    with pytest.raises(ValueError):
        _ = solution[0, 0]


def test_loop_over_solution_is_possible(solution_solve):
    solution, _ = solution_solve

    i = 0
    for i, sol in zip(range(2 * len(solution)), solution):
        assert isinstance(sol, type(solution))

    assert i == len(solution) - 1


# Maybe this test should be in a different test suite, but it does not really matter...
def test_marginal_nth_derivative_of_solution(solution_solve):
    solution, _ = solution_solve

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = solution.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with pytest.raises(ValueError):
        solution.marginals.marginal_nth_derivative(100)
