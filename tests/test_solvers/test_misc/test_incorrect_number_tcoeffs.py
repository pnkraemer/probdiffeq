"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""

from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solvers, strategies


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
    prior = components.ibm_adaptive(num_derivatives=n)

    ts0 = components.correction_ts0()  # irrelevant

    for strategy in [
        strategies.filter_adaptive,
        strategies.smoother_adaptive,
        strategies.fixedpoint_adaptive,
    ]:
        solver = solvers.solver(strategy(prior, ts0))
        output_scale = np.ones_like(impl.prototypes.output_scale())
        with testing.raises(ValueError):
            _ = solver.initial_condition(
                tcoeffs_wrong_length, output_scale=output_scale
            )
