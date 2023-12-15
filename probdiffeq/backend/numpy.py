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


def diff_along_axis(arr, /, *, axis):
    return jnp.diff(arr, axis=axis)


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


def ones(shape, /):
    return jnp.ones(shape)


def ones_like(arr, /):
    return jnp.ones_like(arr)


def zeros_like(arr, /):
    return jnp.zeros_like(arr)


def inf():
    return jnp.inf


def sqrt(arr, /):
    return jnp.sqrt(arr)


def exp(arr, /):
    return jnp.exp(arr)


def eye(n, m=None, /):
    return jnp.eye(n, M=m)


def save(path, arr, /):
    return jnp.save(path, arr)


def load(path, /):
    return jnp.load(path)


def allclose(a, b, *, atol=1e-8, rtol=1e-5):
    return jnp.allclose(a, b, atol=atol, rtol=rtol)


def stack(list_of_arrays, /):
    return jnp.stack(list_of_arrays)


def any(arr, /):  # noqa: A001
    return jnp.any(arr)


def all(arr, /):  # noqa: A001
    return jnp.all(arr)


def sum(arr, /):  # noqa: A001
    return jnp.sum(arr)


def logical_not(a, /):
    return jnp.logical_not(a)


def isinf(arr, /):
    return jnp.isinf(arr)


def isnan(arr, /):
    return jnp.isnan(arr)


def linspace(start, stop, *, num=50, endpoint=True):
    return jnp.linspace(start, stop, num=num, endpoint=endpoint)
