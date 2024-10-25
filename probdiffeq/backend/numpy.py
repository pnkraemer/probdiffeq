"""NumPy-like API."""

import jax.lax
import jax.numpy as jnp


def factorial(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))


def arange(start, stop, *, step=1):
    return jnp.arange(start, stop, step=step)


def ndim(arr, /):
    return jnp.ndim(arr)


def shape(arr, /):
    return jnp.shape(arr)


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


def reshape(arr, /, new_shape, order="C"):
    return jnp.reshape(arr, new_shape, order=order)


def flip(arr, /):
    return jnp.flip(arr)


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


def zeros(shape, /):
    return jnp.zeros(shape)


def empty(shape, /):
    return jnp.empty(shape)


def empty_like(arr, /):
    return jnp.empty_like(arr)


def searchsorted(a, b, /):
    return jnp.searchsorted(a, b)


def block(list_of_arrays, /):
    return jnp.block(list_of_arrays)


def block_diag(list_of_arrays, /):
    return jax.scipy.linalg.block_diag(*list_of_arrays)


def ones_like(arr, /):
    return jnp.ones_like(arr)


def zeros_like(arr, /):
    return jnp.zeros_like(arr)


def inf():
    return jnp.inf


def pi():
    return jnp.pi


def sqrt(arr, /):
    return jnp.sqrt(arr)


def log(arr, /):
    return jnp.log(arr)


def power(arr, powers):
    return jnp.power(arr, powers)


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


def transpose(arr, /, *, axes):
    return jnp.transpose(arr, axes=axes)


def hstack(list_of_arrays, /):
    return jnp.hstack(list_of_arrays)


def kron(a, b, /):
    return jnp.kron(a, b)


def tile(a, num):
    return jnp.tile(a, num)


def repeat(a, num):
    return jnp.repeat(a, num)


def prod_along_axis(arr, /, *, axis):
    return jnp.prod(arr, axis=axis)


def einsum(how, *operands):
    return jnp.einsum(how, *operands)


def meshgrid(*mesh):
    return jnp.meshgrid(*mesh)


def any(arr, /):  # noqa: A001
    return jnp.any(arr)


def all(arr, /):  # noqa: A001
    return jnp.all(arr)


def sum(arr, /):  # noqa: A001
    return jnp.sum(arr)


def logical_not(a, /):
    return jnp.logical_not(a)


def logical_and(a, b, /):
    return jnp.logical_and(a, b)


def isinf(arr, /):
    return jnp.isinf(arr)


def isnan(arr, /):
    return jnp.isnan(arr)


def linspace(start, stop, *, num=50, endpoint=True):
    return jnp.linspace(start, stop, num=num, endpoint=endpoint)
