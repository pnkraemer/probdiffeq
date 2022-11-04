"""Tests for IVP solvers."""
import jax.numpy as jnp
import pytest


def test_solve_computes_correct_terminal_value(
    reference_terminal_values, solution_solve, tolerances
):
    atol, rtol = tolerances
    t_ref, u_ref = reference_terminal_values

    assert jnp.allclose(solution_solve.t[-1], t_ref)
    assert jnp.allclose(solution_solve.u[-1], u_ref, atol=atol, rtol=rtol)


def test_solution_is_iterable(solution_solve):
    assert isinstance(solution_solve[0], type(solution_solve))
    assert len(solution_solve) == len(solution_solve.t)


def test_getitem_raises_error_for_nonbatched_solutions(solution_solve):

    # __getitem__ only works for batched solutions.
    with pytest.raises(ValueError):
        _ = solution_solve[0][0]
    with pytest.raises(ValueError):
        _ = solution_solve[0, 0]


def test_loop_over_solution_is_possible(solution_solve):

    for i, sol in zip(range(2 * len(solution_solve)), solution_solve):
        assert isinstance(sol, type(solution_solve))

    assert i == len(solution_solve) - 1
