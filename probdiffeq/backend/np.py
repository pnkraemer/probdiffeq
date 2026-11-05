"""NumPy-like API."""

import jax.lax
import jax.numpy as jnp
import jax.scipy.special


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


def amax(a, /):
    return jnp.amax(a)


def amin(a, /):
    return jnp.amin(a)


def where(cond, /, if_true, if_false):
    return jnp.where(cond, if_true, if_false)


def abs(arr, /):  # noqa: A001
    return jnp.abs(arr)


def argmin(arr, /):
    return jnp.argmin(arr)


def diff(arr, /, axis=-1):
    return jnp.diff(arr, axis=axis)


def reshape(arr, /, new_shape, order="C"):
    return jnp.reshape(arr, new_shape, order=order)


def flip(arr, /, axis=None):
    return jnp.flip(arr, axis=axis)


def asarray(x, /, dtype=None):
    return jnp.asarray(x, dtype=dtype)


def squeeze(arr, /):
    return jnp.squeeze(arr)


def sign(arr, /):
    return jnp.sign(arr)


def atleast_1d(arr, /):
    return jnp.atleast_1d(arr)


def finfo_eps(arr_or_dtype, /):
    return jnp.finfo(arr_or_dtype).eps


def concatenate(list_of_arrays, /, axis=0):
    return jnp.concatenate(list_of_arrays, axis=axis)


def ones(shape, /, dtype=None):
    return jnp.ones(shape, dtype=dtype)


def zeros(shape, /, dtype=None):
    return jnp.zeros(shape, dtype=dtype)


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


def log2(arr, /):
    return jnp.log2(arr)


def ceil(arr, /):
    return jnp.ceil(arr)


def power(arr, powers):
    return jnp.power(arr, powers)


def exp(arr, /):
    return jnp.exp(arr)


def eye(n, m=None, /, dtype=None):
    return jnp.eye(n, M=m, dtype=dtype)


def save(path, arr, /):
    return jnp.save(path, arr, allow_pickle=True)


def load(path, /):
    return jnp.load(path, allow_pickle=True)


def stack(list_of_arrays, /, *, axis=0):
    return jnp.stack(list_of_arrays, axis=axis)


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


def cumsum(arr, /):
    return jnp.cumsum(arr)


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


def tril(arr, /):
    return jnp.tril(arr)


def triu(arr, /):
    return jnp.triu(arr)


def dtype(arr_or_dtype, /):
    return jnp.dtype(arr_or_dtype)


@jax.custom_jvp
def hypot(x1, x2, /):
    return jnp.hypot(x1, x2)


@hypot.defjvp
def _hypot_jvp(primals, tangents):
    """Evaluate the JVP of hypot, with right-sided derivatives at the origin.

    At the origin, each partial derivative is 1, like grad(abs)(0) == 1.
    Elsewhere, the partial derivatives are x_i / hypot(x1, x2).
    """
    x1, x2 = primals
    dx1, dx2 = tangents
    r = jnp.hypot(x1, x2)
    is_positive = r > 0
    r_safe = jnp.where(is_positive, r, 1.0)
    u1 = jnp.where(is_positive, x1 / r_safe, 1.0)
    u2 = jnp.where(is_positive, x2 / r_safe, 1.0)
    return r, u1 * dx1 + u2 * dx2


def cos(x, /):
    return jnp.cos(x)


def mean(arr, /, axis=None, keepdims=False):
    return jnp.mean(arr, axis=axis, keepdims=keepdims)


def std(arr, /, *, axis=None, ddof=0):
    return jnp.std(arr, ddof=ddof, axis=axis)


def comb(N, k):
    # N choose k
    return jax.scipy.special.comb(N, k)
