"""Linearisation."""

import jax


def ts1(fn, m):
    """Linearise a function with a first-order Taylor series."""
    b, jvp_fn = jax.linearize(fn, m)
    return jvp_fn, (b - jvp_fn(m),)


def ts1_matrix(fn, m):
    """Linearise a function with a first-order Taylor series.

    Same as ts1(), but Jacobians are (dense) matrices instead of callables.
    """
    H = jax.jacfwd(fn)(m)
    b = fn(m)
    return H, (b - H @ m,)
