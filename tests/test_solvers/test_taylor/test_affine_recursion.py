"""Tests for the affine recursion."""

import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.solvers.taylor import affine, autodiff


@testing.parametrize("num", [1, 2, 4])
def test_affine_recursion(num, num_derivatives_max=5):
    """The approximation should coincide with the reference."""
    f, init, solution = _affine_problem(num_derivatives_max)

    derivatives = affine.affine_recursion(f, init, num=num)

    # check shape
    assert len(derivatives) == len(init) + num

    # check values
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)


def _affine_problem(n):
    A = jnp.eye(2) * jnp.arange(1.0, 5.0).reshape((2, 2))
    b = jnp.arange(6.0, 8.0)

    def vf(x, /):
        return A @ x + b

    init = (jnp.arange(9.0, 11.0),)

    solution = autodiff.taylor_mode(vf, init, num=n)
    return vf, init, solution
