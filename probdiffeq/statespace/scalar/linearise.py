"""Linearisation."""

import jax


def ts1(fn, m):
    """Linearise a function with a first-order Taylor series."""
    b, jvp_fn = jax.linearize(fn, m)
    return jvp_fn, (b - jvp_fn(m),)
