"""Linear-algebra routines."""

import jax.numpy as jnp
import jax.scipy.linalg


def qr_r(arr, /):
    return jnp.linalg.qr(arr, mode="r")


# All Cholesky factors are lower-triangular by default


def cholesky_factor(arr, /):
    return jnp.linalg.cholesky(arr)


# All Cholesky factors are lower-triangular by default


def cholesky_solve(arr, rhs, /):
    return jax.scipy.linalg.cho_solve((arr, True), rhs)


def vector_norm(arr, /, *, order=None):
    return jnp.linalg.norm(arr, ord=order)


def matrix_norm(arr, /, *, order=None):
    return jnp.linalg.norm(arr, ord=order)


def solve_triangular(matrix, rhs, /, *, trans=0, lower=False):
    return jax.scipy.linalg.solve_triangular(matrix, rhs, trans=trans, lower=lower)


def inv(matrix, /):
    return jnp.linalg.inv(matrix)
