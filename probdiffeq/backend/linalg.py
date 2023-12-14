"""Linear-algebra routines."""

import jax.numpy as jnp


def qr_r(arr, /):
    return jnp.linalg.qr(arr, mode="r")
