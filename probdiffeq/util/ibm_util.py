"""Integrated Brownian motion (IBM) utilities."""

import jax
import jax.numpy as jnp


def system_matrices_1d(num_derivatives, output_scale):
    """Construct the IBM system matrices."""
    x = jnp.arange(num_derivatives + 1)

    A_1d = jnp.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = jnp.flip(_hilbert(x))
    return A_1d, output_scale * jnp.linalg.cholesky(Q_1d)


def preconditioner_diagonal(dt, *, scales, powers):
    """Construct the diagonal IBM preconditioner."""
    dt_abs = jnp.abs(dt)
    scaling_vector = jnp.power(dt_abs, powers) / scales
    scaling_vector_inv = jnp.power(dt_abs, -powers) * scales
    return scaling_vector, scaling_vector_inv


def preconditioner_prepare(*, num_derivatives):
    powers = jnp.arange(num_derivatives, -1.0, -1.0)
    scales = _factorial(powers)
    powers = powers + 0.5
    return jax.tree_util.Partial(preconditioner_diagonal, scales=scales, powers=powers)


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a, /):
    return _batch_gram(_binom)(a[:, None], a[None, :])


def _batch_gram(k, /):
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    return jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)


def _binom(n, k):
    return _factorial(n) / (_factorial(n - k) * _factorial(k))


def _factorial(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
