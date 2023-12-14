"""NumPy-like API."""

import jax.lax


def factorial(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
