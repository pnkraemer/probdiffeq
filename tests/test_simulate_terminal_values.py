"""Tests for IVP solvers."""

import jax.numpy as jnp


def test_terminal_values_simulated_correctly(
    reference_terminal_values, solution_terminal_values, tolerances
):
    t_ref, u_ref = reference_terminal_values
    atol, rtol = tolerances

    assert solution_terminal_values.t == t_ref
    assert jnp.allclose(solution_terminal_values.u, u_ref, atol=atol, rtol=rtol)
