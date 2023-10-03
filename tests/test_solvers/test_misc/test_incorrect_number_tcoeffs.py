"""Tests for miscellaneous edge cases.

Place all tests that have no better place here.
"""
import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import filters, fixedpoint, smoothers
from probdiffeq.solvers.strategies.components import corrections, priors


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
    prior = priors.ibm_adaptive(num_derivatives=n)

    ts0 = corrections.ts0()  # irrelevant

    for strategy in [
        filters.filter_adaptive,
        smoothers.smoother_adaptive,
        fixedpoint.fixedpoint_adaptive,
    ]:
        solver = uncalibrated.solver(strategy(prior, ts0))
        output_scale = jnp.ones_like(impl.prototypes.output_scale())
        with testing.raises(ValueError):
            _ = solver.initial_condition(
                tcoeffs_wrong_length, output_scale=output_scale
            )
