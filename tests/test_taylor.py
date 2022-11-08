"""Tests for Taylor series approximations / initialisation functions."""

import diffeqzoo.ivps
import pytest_cases

from odefilter import taylor


@pytest_cases.case(tags=["first", "higher"])
def case_forward_mode():
    return taylor.taylor_mode_fn


@pytest_cases.case(tags=["first", "higher"])
def case_taylor_mode():
    return taylor.forward_mode_fn


@pytest_cases.case(tags=["first"])
def case_runge_kutta_starter():
    return taylor.make_runge_kutta_starter_fn()


@pytest_cases.case(tags=["first"])
def pb_three_body_first():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.three_body_restricted_first_order()

    def vf(u, *, t, p):
        return f(u, *p)

    return vf, (u0,), t0, f_args


@pytest_cases.case(tags=["higher"])
def pb_van_der_pol_second_order():
    f, (u0, du0), (t0, _), f_args = diffeqzoo.ivps.van_der_pol()

    def vf(u, du, *, t, p):
        return f(u, du, *p)

    return vf, (u0, du0), t0, f_args


@pytest_cases.parametrize_with_cases(
    "fn", cases=".", prefix="case_", has_tag=["higher"]
)
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["higher"])
@pytest_cases.parametrize("num", [1, 2])
def test_initialised_correct_shape_higher_order(fn, pb, num):
    f, init, t0, params = pb
    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )

    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape


@pytest_cases.parametrize_with_cases("fn", cases=".", prefix="case_", has_tag=["first"])
@pytest_cases.parametrize_with_cases("pb", cases=".", prefix="pb_", has_tag=["first"])
@pytest_cases.parametrize("num", [1, 2])
def test_initialised_correct_shape_first_order(fn, pb, num):
    f, init, t0, params = pb

    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )
    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
