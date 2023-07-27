"""Tests for BVP solver."""

import diffeqzoo.bvps
import jax.numpy as jnp

from probdiffeq import bvpsolve


def test_bvpsolve():
    """Solve a second-order, scalar, linear, separable BVP."""

    vf, (g0, g1), (t0, t1), params = diffeqzoo.bvps.pendulum()

    grid = jnp.linspace(t0, t1, num=10)
    solution = bvpsolve.solve_fixed_grid(vf, bcond=(g0, g1), grid=grid)

    print(solution)

    assert False
