"""Tests for specific edge cases.

Place all tests that have no better place here.
"""
import pytest
import pytest_cases

from probdiffeq import solution_routines, test_util


@pytest_cases.parametrize("incr", [1, -1])
@pytest_cases.parametrize("n", [2])
def test_incorrect_number_of_taylor_coefficients_init(incr, n):
    solver = test_util.generate_solver(num_derivatives=n)
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant

    init_fn = solver.strategy.implementation.extrapolation.init_hidden_state
    with pytest.raises(ValueError):
        init_fn(taylor_coefficients=tcoeffs_wrong_length)


# Assert correct behaviour of the solution object (raise error for wrong slicing etc.)


@pytest_cases.fixture(name="dummy_solution")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def fixture_dummy_solution(ode_problem):
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
    return solution


def test_solution_is_iterable(dummy_solution):
    assert isinstance(dummy_solution[0], type(dummy_solution))
    assert len(dummy_solution) == len(dummy_solution.t)


def test_getitem_raises_error_for_nonbatched_solutions(dummy_solution):
    """__getitem__ only works for batched solutions."""
    with pytest.raises(ValueError):
        _ = dummy_solution[0][0]
    with pytest.raises(ValueError):
        _ = dummy_solution[0, 0]


def test_loop_over_solution_is_possible(dummy_solution):
    i = 0
    for i, sol in zip(range(2 * len(dummy_solution)), dummy_solution):
        assert isinstance(sol, type(dummy_solution))

    assert i == len(dummy_solution) - 1


# Maybe this test should be in a different test suite, but it does not really matter...
def test_marginal_nth_derivative_of_solution(dummy_solution):
    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = dummy_solution.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == dummy_solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with pytest.raises(ValueError):
        dummy_solution.marginals.marginal_nth_derivative(100)
