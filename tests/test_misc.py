"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""
import pytest
import pytest_cases

from probdiffeq import test_util


@pytest_cases.parametrize("incr", [1, -1])
@pytest_cases.parametrize("n", [2])
def test_incorrect_number_of_taylor_coefficients_init(incr, n):
    solver = test_util.generate_solver(num_derivatives=n)
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant

    init_fn = solver.strategy.implementation.extrapolation.init_hidden_state
    with pytest.raises(ValueError):
        init_fn(taylor_coefficients=tcoeffs_wrong_length)
