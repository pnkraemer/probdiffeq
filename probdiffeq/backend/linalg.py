"""Linear-algebra routines."""

import jax.numpy as jnp
import jax.scipy.linalg


def qr_r(arr, /):
    return jnp.linalg.qr(arr, mode="r")


def cholesky_lower(arr, /):
    return jnp.linalg.cholesky(arr)


def vector_norm(arr, /, *, order=None):
    return jnp.linalg.norm(arr, ord=order)


def matrix_norm(arr, /, *, order=None):
    return jnp.linalg.norm(arr, ord=order)


def solve_triangular(matrix, rhs, /, *, trans=0, lower=False):
    return jax.scipy.linalg.solve_triangular(matrix, rhs, trans=trans, lower=lower)
