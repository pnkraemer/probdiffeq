"""Test that the control_flow can be updated by a user."""

from probdiffeq.backend import control_flow
from probdiffeq.backend import numpy as np


def test_overwrite_scan_func():
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

    def scan_that_adds_1(step, init, xs):
        return jax.lax.scan(step, init=init + 1, xs=xs)

    control_flow.overwrite_scan_func(scan_that_adds_1)

    final, outputs = control_flow.scan(cumsum_step, init=0.0, xs=xs)
    assert np.allclose(final, sum_total + 1.0)
    assert np.allclose(outputs, cumsum_total + 1.0)


def test_overwrite_while_loop_func():
    def counter_step(x):
        return x[0] + 1, x[1]

    index, value = control_flow.while_loop(lambda s: s[0] < 10, counter_step, (0, 0.0))
    assert np.allclose(index, 10)
    assert np.allclose(value, 0.0)

    # Direct import;
    # Do not use probdiffeq.backend since otherwise we recurse
    import jax.lax

    def while_loop_that_adds_1(cond, body, init):
        idx, val = init
        init_new = (idx, val + 1.0)
        return jax.lax.while_loop(cond, body, init_new)

    control_flow.overwrite_while_loop_func(while_loop_that_adds_1)
    index, value = control_flow.while_loop(lambda s: s[0] < 10, counter_step, (0, 0.0))
    assert np.allclose(index, 10)
    assert np.allclose(value, 1.0)  # instead of 0.
