"""Test that the control_flow can be updated by a user."""

from probdiffeq.backend import control_flow
from probdiffeq.backend import numpy as np


def test_scan():
    def cumsum_step(carry, x):
        res = carry + x
        return res, res

    xs = np.arange(1.0, 11.0, step=2.0)
    sum_total = 25
    cumsum_total = np.asarray([1.0, 4.0, 9.0, 16.0, 25])

    final, outputs = control_flow.scan(cumsum_step, init=0.0, xs=xs)
    assert np.allclose(final, sum_total)
    assert np.allclose(outputs, cumsum_total)

    # Direct import;
    # Do not use probdiffeq.backend since otherwise we recurse
    import jax.lax

    def scan_that_adds_1(*args, init, **kwargs):
        return jax.lax.scan(*args, init=init + 1, **kwargs)

    control_flow.overwrite_func_scan(scan_that_adds_1)

    final, outputs = control_flow.scan(cumsum_step, init=0.0, xs=xs)
    assert np.allclose(final, sum_total + 1.0)
    assert np.allclose(outputs, cumsum_total + 1.0)
