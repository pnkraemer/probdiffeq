"""Tests for initialisation functions."""

import jax
import pytest_cases
from diffeqzoo import ivps as ivpzoo

from odefilter import taylor


@pytest_cases.case
def init_forward_mode():
    return taylor.taylor_mode_fn


@pytest_cases.case
def init_taylor_mode():
    return taylor.forward_mode_fn


@pytest_cases.case
def problem_three_body_first_order():
    f, u0, _, f_args = ivpzoo.three_body_restricted_first_order()

    @jax.jit
    def vf(u):
        return f(u, *f_args)

    return vf, (u0,)


@pytest_cases.case
def problem_van_der_pol_second_order():
    f, (u0, du0), _, f_args = ivpzoo.van_der_pol()

    @jax.jit
    def vf(u, du):
        return f(u, du, *f_args)

    return vf, (u0, du0)


@pytest_cases.parametrize_with_cases("init_fn", cases=".", prefix="init_")
@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize("num", [1, 2])
def test_init(init_fn, problem, num):
    f, init = problem
    derivatives = init_fn(vector_field=f, initial_values=init, num=num)

    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
