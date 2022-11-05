"""Tests for solving IVPs on adaptive grids."""
import jax.numpy as jnp
import pytest


def test_solve_computes_correct_terminal_value(
    reference_terminal_values, solution_solve, tolerances
):
    atol, rtol = tolerances
    t_ref, u_ref = reference_terminal_values
    solution, _ = solution_solve

    assert jnp.allclose(solution.t[-1], t_ref)
    assert jnp.allclose(solution.u[-1], u_ref, atol=atol, rtol=rtol)


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

    for i, sol in zip(range(2 * len(solution)), solution):
        assert isinstance(sol, type(solution))

    assert i == len(solution) - 1
