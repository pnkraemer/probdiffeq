"""Tests for inexact approximations for first-order problems."""
import diffeqzoo.ivps
import jax.numpy as jnp

from probdiffeq import taylor
from probdiffeq.backend import testing


@testing.case()
def case_runge_kutta_starter():
    return taylor.make_runge_kutta_starter_fn()


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, *, t, p):  # pylint: disable=unused-argument
        return f(u, *p)

    solution = jnp.load("./tests/test_taylor/data/three_body_first_solution.npy")
    return (vf, (u0,), t0, f_args), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 3])
def test_initialised_correct_shape(pb_with_solution, taylor_fun, num):
    (f, init, t0, params), _solution = pb_with_solution
    derivatives = taylor_fun(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )
    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
