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


def diff(arr, /):
    return jnp.diff(arr)


def asarray(x, /):
    return jnp.asarray(x)


def squeeze(arr, /):
    return jnp.squeeze(arr)


def squeeze_along_axis(arr, /, *, axis):
    return jnp.squeeze(arr, axis=axis)


def atleast_1d(arr, /):
    return jnp.atleast_1d(arr)


def concatenate(list_of_arrays, /):
    return jnp.concatenate(list_of_arrays)
