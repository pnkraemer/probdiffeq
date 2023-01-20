"""Tests for Taylor series approximations / initialisation functions."""

import diffeqzoo.ivps
import jax.numpy as jnp
import pytest_cases

from probdiffeq import taylor


@pytest_cases.case(tags=["first", "higher", "exact"])
def case_forward_mode():
    return taylor.forward_mode_fn


@pytest_cases.case(tags=["first", "higher", "exact"])
def case_taylor_mode():
    return taylor.taylor_mode_fn


@pytest_cases.case(tags=["first", "exact"])
def case_taylor_mode_doubling():
    return taylor.taylor_mode_doubling_fn


@pytest_cases.case(tags=["first"])
def case_runge_kutta_starter():
    return taylor.make_runge_kutta_starter_fn()


@pytest_cases.fixture(name="num_derivatives_max")
def fixture_num_derivatives_max():
    return 3


@pytest_cases.case(tags=["first"])
def pb_three_body_first(num_derivatives_max):
    f, u0, (t0, _), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, *, t, p):  # pylint: disable=unused-argument
        return f(u, *p)

    solution = taylor.taylor_mode_fn(
        vector_field=vf,
        initial_values=(u0,),
        num=num_derivatives_max,
        t=t0,
        parameters=f_args,
    )
    return vf, (u0,), t0, f_args, solution


@pytest_cases.case(tags=["higher"])
def pb_van_der_pol_second_order(num_derivatives_max):
    f, (u0, du0), (t0, _), f_args = diffeqzoo.ivps.van_der_pol()

    def vf(u, du, *, t, p):  # pylint: disable=unused-argument
        return f(u, du, *p)

    solution = taylor.taylor_mode_fn(
        vector_field=vf,
        initial_values=(u0, du0),
        num=num_derivatives_max,
        t=t0,
        parameters=f_args,
    )
    return vf, (u0, du0), t0, f_args, solution


@pytest_cases.parametrize_with_cases(
    "fn", cases=".", prefix="case_", has_tag=["higher"]
)
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["higher"])
@pytest_cases.parametrize("num", [1, 3])
def test_initialised_correct_shape_higher_order(fn, pb, num):
    f, init, t0, params, _ = pb
    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )

    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape


@pytest_cases.parametrize_with_cases("fn", cases=".", prefix="case_", has_tag=["first"])
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["first"])
@pytest_cases.parametrize("num", [1, 3])
def test_initialised_correct_shape_first_order(fn, pb, num):
    f, init, t0, params, _ = pb
    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )
    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape


@pytest_cases.parametrize_with_cases(
    "fn", cases=".", prefix="case_", has_tag=["first", "exact"]
)
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["first"])
@pytest_cases.parametrize("num", [1, 3])
def test_initialised_exactly_first(fn, pb, num):
    """The approximation should coincide with the reference."""
    f, init, t0, params, solution = pb

    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)


@pytest_cases.parametrize_with_cases(
    "fn", cases=".", prefix="case_", has_tag=["higher", "exact"]
)
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["higher"])
@pytest_cases.parametrize("num", [1, 2, 4])
def test_initialised_exactly_higher(fn, pb, num):
    """The approximation should coincide with the reference."""
    f, init, t0, params, solution = pb

    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )
    for dy, dy_ref in zip(derivatives, solution):
        assert jnp.allclose(dy, dy_ref)


@pytest_cases.parametrize("num", [1, 2, 4])
def test_affine_recursion(num_derivatives_max, num):
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
