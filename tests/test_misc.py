"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""
from probdiffeq.backend import testing
from probdiffeq.statespace.iso import extra


@testing.parametrize("incr", [1, -1])
@testing.parametrize("n", [2])
def test_incorrect_number_of_taylor_coefficients_init(incr, n):
    """Assert that a specific ValueError is raised.

    Specifically:
    A ValueError must be raised if the number of Taylor coefficients
    passed to *IBM.init_state_space_var() does not match the `num_derivatives`
    attribute of the extrapolation model.
    """
    tcoeffs_wrong_length = [None] * (n + 1 + incr)  # 'None' bc. values irrelevant
    factory = extra.ibm_factory(num_derivatives=n)
    for impl in [factory.filter(), factory.smoother(), factory.fixedpoint()]:
        with testing.raises(ValueError):
            _ = impl.solution_from_tcoeffs(tcoeffs_wrong_length)
