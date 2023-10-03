"""Test the exactness of differentiation-based routines on first-order problems."""

import diffeqzoo.ivps
import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.solvers.taylor import autodiff


@testing.case()
def case_forward_mode():
    return autodiff.forward_mode


@testing.case()
def case_taylor_mode():
    return autodiff.taylor_mode


@testing.case()
def case_taylor_mode_doubling():
    return autodiff.taylor_mode_doubling


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, /):
        return f(u, *f_args)

    solution = jnp.load(
        "./tests/test_solvers/test_taylor/data/three_body_first_solution.npy"
    )
    return (vf, (u0,)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 3])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)
