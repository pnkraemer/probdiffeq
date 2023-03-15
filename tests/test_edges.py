"""Tests for specific edge cases.

Place all tests that have no better place here.
"""

import pytest
import pytest_cases


@pytest_cases.parametrize_with_cases("solver", cases=".solver_cases")
@pytest_cases.parametrize("incr", [1, -1])
def test_incorrect_number_of_taylor_coefficients_init(solver, incr):
    n = solver.strategy.implementation.extrapolation.num_derivatives
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant

    init_fn = solver.strategy.implementation.extrapolation.init_hidden_state
    with pytest.raises(ValueError):
        init_fn(taylor_coefficients=tcoeffs_wrong_length)
