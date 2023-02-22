"""Integrated Brownian motion (IBM) utilities."""

import jax
import jax.numpy as jnp


def system_matrices_1d(*, num_derivatives):
    """Construct the IBM system matrices."""
    x = jnp.arange(num_derivatives + 1)

    A_1d = jnp.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = jnp.flip(_hilbert(x))
    return A_1d, jnp.linalg.cholesky(Q_1d)


def preconditioner_diagonal(*, dt, num_derivatives):
    """Construct the diagonal IBM preconditioner."""
    powers = jnp.arange(num_derivatives, -1, -1)

    scales = _factorial(powers)
    powers = powers + 0.5

    scaling_vector = (jnp.abs(dt) ** powers) / scales
    scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales

    return scaling_vector, scaling_vector_inv


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    k_vmapped_xy = jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)
    return k_vmapped_xy


def _binom(n, k):
    return _factorial(n) / (_factorial(n - k) * _factorial(k))


def _factorial(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
