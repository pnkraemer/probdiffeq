"""Tests for solving IVPs for the terminal value."""

import jax.numpy as jnp


def test_terminal_values_simulated_correctly(
    reference_terminal_values, solution_terminal_values, tolerances
):
    t_ref, u_ref = reference_terminal_values
    atol, rtol = tolerances
    solution, _ = solution_terminal_values

    assert solution.t == t_ref
    assert jnp.allclose(solution.u, u_ref, atol=atol, rtol=rtol)
