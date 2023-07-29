"""Test the exactness of differentiation-based routines on first-order problems."""

import diffeqzoo.ivps
import jax.numpy as jnp

from probdiffeq import taylor
from probdiffeq.backend import testing


@testing.case()
def case_forward_mode():
    return taylor.forward_mode_fn


@testing.case()
def case_taylor_mode():
    return taylor.taylor_mode_fn


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    f, (u0, du0), (t0, _), f_args = diffeqzoo.ivps.van_der_pol()

    def vf(u, du, *, t, p):  # pylint: disable=unused-argument
        return f(u, du, *p)

    solution = jnp.load("./tests/test_taylor/data/van_der_pol_second_solution.npy")
    return (vf, (u0, du0), t0, f_args), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 3])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init, t0, params), solution = pb_with_solution

    derivatives = taylor_fun(
        vector_field=f, initial_values=init, t=t0, parameters=params, num=num
    )
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)
