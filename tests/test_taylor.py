"""Tests for initialisation functions."""

import jax
import jax.tree_util
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
    f, u0, _, f_args = ivpzoo.three_body_restricted_first_order()

    def vector_field(u, *, p):
        return f(u, *p)

    vf = jax.tree_util.Partial(vector_field, p=f_args)

    return vf, (u0,)


@pytest_cases.case
def problem_van_der_pol_second_order():
    f, (u0, du0), _, f_args = ivpzoo.van_der_pol()

    def vector_field(u, du, *, p):
        return f(u, du, *p)

    # Emulate how it will be called in the solvers.
    vf = jax.tree_util.Partial(vector_field, p=f_args)

    return vf, (u0, du0)


@pytest_cases.parametrize_with_cases("fn", cases=".", prefix="case_")
@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize("num", [1, 2])
def test_init(fn, problem, num):
    f, init = problem
    derivatives = fn(vector_field=f, initial_values=init, num=num)

    assert len(derivatives) == len(init) + num
    assert derivatives[0].shape == init[0].shape
