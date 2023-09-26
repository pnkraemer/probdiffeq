"""Tests for inexact approximations for first-order problems."""
import diffeqzoo.ivps
import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers.taylor import autodiff, estim


@testing.case()
def case_runge_kutta_starter():
    if impl.impl_name != "isotropic":
        testing.skip(reason="Runge-Kutta starters currently require isotropic SSMs.")
    return estim.make_runge_kutta_starter(dt=0.01)


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()

    def vf(u, /):
        return f(u, *f_args)

    solution = autodiff.taylor_mode(vf, (u0,), num=3)
    return (vf, (u0,), t0), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 4])
def test_initialised_correct_shape_and_values(pb_with_solution, taylor_fun, num):
    (f, init, t0), _solution = pb_with_solution
    derivatives = taylor_fun(lambda y, _t: f(y), init, t=t0, num=num)
    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
    for expected, received in zip(derivatives, _solution):
        # demand at least ~10% accuracy to warn about the most obvious bugs
        assert jnp.allclose(expected, received, rtol=1e-1), (expected, received)
