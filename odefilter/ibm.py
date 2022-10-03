"""Integrated Brownian motion."""

from functools import partial

import jax.lax
import jax.numpy as jnp


def system_matrices_1d(*, num_derivatives):
    x = jnp.arange(num_derivatives + 1)

    A_1d = jnp.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = jnp.flip(_hilbert(x))
    return A_1d, jnp.linalg.cholesky(Q_1d)


def preconditioner(*, dt, num_derivatives):
    p, p_inv = preconditioner_diagonal(dt=dt, num_derivatives=num_derivatives)
    return jnp.diag(p), jnp.diag(p_inv)


def preconditioner_diagonal(*, dt, num_derivatives):
    powers = jnp.arange(num_derivatives, -1, -1)

    scales = _factorial(powers)
    powers = powers + 0.5

    scaling_vector = (jnp.abs(dt) ** powers) / scales
    scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales

    return scaling_vector, scaling_vector_inv


@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
def preconditioner_diagonal_batched(dts, num_derivatives):
    """Computes the diagonal preconditioner, but for a number of time-steps at once."""
    return preconditioner_diagonal(dt=dts, num_derivatives=num_derivatives)


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a):
    return _binom(a[:, None], a[None, :])


def _batch_gram(k):
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    k_vmapped_xy = jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)
    return jax.jit(k_vmapped_xy)


@_batch_gram
def _binom(n, k):
    a = _factorial(n)
    b = _factorial(n - k)
    c = _factorial(k)
    return a / (b * c)


def _factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
