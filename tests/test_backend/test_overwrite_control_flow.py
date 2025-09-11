"""Test that the control_flow can be updated by a user."""

from probdiffeq.backend import control_flow, testing
from probdiffeq.backend import numpy as np


def test_overwrite_scan_func():
    def cumsum_step(carry, x):
        res = carry + x
        return res, res

    xs = np.arange(1.0, 11.0, step=2.0)
    sum_total = 25
    cumsum_total = np.asarray([1.0, 4.0, 9.0, 16.0, 25])

    final, outputs = control_flow.scan(cumsum_step, init=0.0, xs=xs)
    assert testing.allclose(final, sum_total)
    assert testing.allclose(outputs, cumsum_total)

    # Direct import;
    # Do not use probdiffeq.backend since otherwise we recurse
    import jax.lax

    def scan_that_adds_1(step, init, xs, reverse, length):
        return jax.lax.scan(step, init=init + 1, xs=xs, reverse=reverse, length=length)

    with control_flow.context_overwrite_scan(scan_that_adds_1):
        final, outputs = control_flow.scan(cumsum_step, init=0.0, xs=xs)
    assert testing.allclose(final, sum_total + 1.0)
    assert testing.allclose(outputs, cumsum_total + 1.0)


def test_overwrite_while_loop_func():
    def counter_step(x):
        return x[0] + 1, x[1]

    index, value = control_flow.while_loop(lambda s: s[0] < 10, counter_step, (0, 0.0))
    assert testing.allclose(index, 10)
    assert testing.allclose(value, 0.0)

    # Direct import;
    # Do not use probdiffeq.backend since otherwise we recurse
    import jax.lax

    # mirror jax.lax.while_loop signature, which may differ
    # from the backend.control_flow.while_loop signature
    def while_loop_that_adds_1(cond_fun, body_fun, init_val):
        idx, val = init_val
        init_new = (idx, val + 1.0)
        return jax.lax.while_loop(cond_fun, body_fun, init_new)

    with control_flow.context_overwrite_while_loop(while_loop_that_adds_1):
        index, value = control_flow.while_loop(
            lambda s: s[0] < 10, counter_step, (0, 0.0)
        )
    assert testing.allclose(index, 10)
    assert testing.allclose(value, 1.0)  # instead of 0.
