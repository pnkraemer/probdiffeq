"""Tests for the affine recursion."""

import jax.numpy as jnp

from probdiffeq import taylor
from probdiffeq.backend import testing


@testing.parametrize("num", [1, 2, 4])
def test_affine_recursion(num, num_derivatives_max=5):
    """The approximation should coincide with the reference."""
    f, init, t0, params, solution = _affine_problem(num_derivatives_max)

    derivatives = taylor.affine_recursion(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )

    # check shape
    assert len(derivatives) == len(init) + num

    # check values
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)


def _affine_problem(n):
    A0 = jnp.eye(2) * jnp.arange(1.0, 5.0).reshape((2, 2))
    b0 = jnp.arange(6.0, 8.0)

    def vf(x, /, *, t, p):  # pylint: disable=unused-argument
        A, b = p
        return A @ x + b

    init = (jnp.arange(9.0, 11.0),)
    t0 = 0.0
    f_args = (A0, b0)

    solution = taylor.taylor_mode_fn(
        vector_field=vf,
        initial_values=init,
        num=n,
        t=t0,
        parameters=f_args,
    )
    return vf, init, t0, f_args, solution
