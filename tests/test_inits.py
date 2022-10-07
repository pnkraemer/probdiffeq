"""Tests for initialisation functions."""

import jax
import pytest_cases
from diffeqzoo import ivps as ivpzoo

from odefilter import inits


@pytest_cases.case
def init_taylor_mode():
    return inits.taylor_mode


@pytest_cases.case
def problem_three_body_second_order():
    f, (u0, du0), _, f_args = ivpzoo.three_body_restricted()
    return lambda u, du: f(u, du, *f_args), (u0, du0)


@pytest_cases.case
def problem_three_body_first_order():
    f, u0, _, f_args = ivpzoo.three_body_restricted_first_order()
    return lambda u: f(u, *f_args), (u0,)


@pytest_cases.case
def problem_van_der_pol_second_order():
    f, (u0, du0), _, f_args = ivpzoo.van_der_pol()
    return lambda u, du: f(u, du, *f_args), (u0, du0)


@pytest_cases.parametrize_with_cases("init_fn", cases=".", prefix="init_")
@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize("num_recursions", [1, 4])
def test_init(init_fn, problem, num_recursions):
    f, init = problem
    derivatives = init_fn(
        vector_field=f, initial_values=init, num_recursions=num_recursions
    )

    assert len(derivatives) == len(init) + num_recursions
    assert derivatives[0].shape == init[0].shape
