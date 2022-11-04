"""Tests for IVP solvers."""

import jax.numpy as jnp


def test_solve_fixed_grid_computes_terminal_values_correctly(
    reference_terminal_values, solution_fixed_grid, tolerances
):
    t_ref, u_ref = reference_terminal_values
    atol, rtol = tolerances

    assert jnp.allclose(solution_fixed_grid.t[-1], t_ref)
    assert jnp.allclose(solution_fixed_grid.u[-1], u_ref, atol=atol, rtol=rtol)
