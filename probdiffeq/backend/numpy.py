"""NumPy-like API."""

import jax.lax
import jax.numpy as jnp


def factorial(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))


def arange(start, stop, *, step=1):
    return jnp.arange(start, stop, step=step)


def ndim(arr, /):
    return jnp.ndim(arr)


def minimum(a, b, /):
    return jnp.minimum(a, b)


def maximum(a, b, /):
    return jnp.maximum(a, b)


def where(cond, /, if_true, if_false):
    return jnp.where(cond, if_true, if_false)


def abs(arr, /):  # noqa: A001
    return jnp.abs(arr)


def finfo_eps(eltype, /):
    return jnp.finfo(eltype).eps
