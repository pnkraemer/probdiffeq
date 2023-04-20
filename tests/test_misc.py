"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""
from probdiffeq import test_util
from probdiffeq.backend import testing


@testing.parametrize("incr", [1, -1])
@testing.parametrize("n", [2])
def test_incorrect_number_of_taylor_coefficients_init(incr, n):
    """Assert that a specific ValueError is raised.

    Specifically:
    A ValueError must be raised if the number of Taylor coefficients
    passed to *IBM.init_state_space_var() does not match the `num_derivatives`
    attribute of the extrapolation model.
    """
    solver = test_util.generate_solver(num_derivatives=n)
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant

    extra = solver.strategy.extrapolation
    for impl in [extra.filter, extra.smoother, extra.fixedpoint]:
        with testing.raises(ValueError):
            _ = impl.solution_from_tcoeffs(tcoeffs_wrong_length)
