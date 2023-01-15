"""Tests for solving IVPs on adaptive grids."""
import jax.numpy as jnp
import pytest


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
