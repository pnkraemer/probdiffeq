"""Tests for solving IVPs for the terminal value."""

import jax.numpy as jnp


def test_terminal_values_simulated_correctly(
    reference_terminal_values, solution_terminal_values, solver_config
):
    t_ref, u_ref = reference_terminal_values
    solution, _ = solution_terminal_values

    assert solution.t == t_ref
    assert jnp.allclose(
        solution.u,
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )
