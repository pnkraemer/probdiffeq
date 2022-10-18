"""Tests for initialisation functions."""

import pytest_cases
from diffeqzoo import ivps as ivpzoo

from odefilter import taylor


@pytest_cases.case
def case_forward_mode():
    return taylor.taylor_mode_fn


@pytest_cases.case
def case_taylor_mode():
    return taylor.forward_mode_fn


@pytest_cases.case
def problem_three_body_first_order():
    f, u0, (t0, _), f_args = ivpzoo.three_body_restricted_first_order()

    def vf(u, *, t, p):
        return f(u, *p)

    return vf, (u0,), t0, f_args


@pytest_cases.case
def problem_van_der_pol_second_order():
    f, (u0, du0), (t0, _), f_args = ivpzoo.van_der_pol()

    def vf(u, du, *, t, p):
        return f(u, du, *p)

    return vf, (u0, du0), t0, f_args


@pytest_cases.parametrize_with_cases("fn", cases=".", prefix="case_")
@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize("num", [1, 2])
def test_init(fn, problem, num):
    f, init, t0, params = problem
    derivatives = fn(
        vector_field=f, initial_values=init, num=num, t=t0, parameters=params
    )

    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
