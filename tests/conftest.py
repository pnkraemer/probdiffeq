"""Test configurations."""

import dataclasses
import functools
from typing import Callable

import jax
import jax.experimental.ode
import jax.numpy as jnp
import pytest_cases

# Solver configurations (for example, tolerances.)
# My attempt at bundling up all those magic save_at grids, tolerances, etc.


@dataclasses.dataclass
class SolverConfiguration:
    atol_solve: float
    rtol_solve: float
    grid_for_fixed_grid_fn: Callable[[float, float], jax.Array]
    grid_for_save_at_fn: Callable[[float, float], jax.Array]

    @property
    def atol_assert(self):
        return 10 * self.atol_solve

    @property
    def rtol_assert(self):
        return 10 * self.rtol_solve


@pytest_cases.fixture(scope="session", name="solver_config")
def fixture_solver_config():
    grid_fn = functools.partial(jnp.linspace, endpoint=True, num=10)
    save_at_fn = functools.partial(jnp.linspace, endpoint=True, num=5)
    return SolverConfiguration(
        atol_solve=1e-5,
        rtol_solve=1e-3,
        grid_for_fixed_grid_fn=grid_fn,
        grid_for_save_at_fn=save_at_fn,
    )
