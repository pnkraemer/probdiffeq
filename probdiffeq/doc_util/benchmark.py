"""Benchmark utils."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probdiffeq import solution_routines


def relative_rmse(*, solution: ArrayLike, atol=1e-5):
    """Relative root mean-squared error."""
    solution = jnp.asarray(solution)

    @jax.jit
    def error_fn(u: ArrayLike, /):
        ratio = (u - solution) / (atol + solution)
        return jnp.linalg.norm(ratio) / jnp.sqrt(ratio.size)

    return error_fn


def probdiffeq_terminal_values():
    def solve_fn(*problem, atol, rtol, **method):
        solution = solution_routines.simulate_terminal_values(
            *problem, atol=atol, rtol=rtol, **method
        )
        return solution.u

    return solve_fn
