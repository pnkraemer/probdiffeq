"""Linear-algebra routines."""

import jax.numpy as jnp


def qr_r(arr, /):
    return jnp.linalg.qr(arr, mode="r")


def cholesky_lower(arr, /):
    return jnp.linalg.cholesky(arr)


def vector_norm(arr, /):
    assert arr.ndim == 1
    return jnp.linalg.norm(arr)


def matrix_norm(arr, /):
    assert arr.ndim == 2
    return jnp.linalg.norm(arr)
