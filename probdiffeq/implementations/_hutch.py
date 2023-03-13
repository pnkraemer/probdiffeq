"""Hutchinson-style trace and diagonal estimation."""

import jax
import jax.numpy as jnp


def trace(*args, matvec_fn, **kwargs):
    """Estimate the trace of a matrix stochastically."""

    def Q(x):
        return jnp.dot(x, matvec_fn(x))

    return _hutchinson(Q, *args, **kwargs)


def diagonal(*args, matvec_fn, **kwargs):
    """Estimate the diagonal of a matrix stochastically."""

    def Q(x):
        return x * matvec_fn(x)

    return _hutchinson(Q, *args, **kwargs)


def _hutchinson(*args, batch_keys, **kwargs):
    """Hutchinson-style trace estimation."""

    @jax.jit
    def f(key):
        return _hutchinson_batch(*args, key=key, **kwargs)

    # Compute batches sequentially to reduce memory.
    batch_keys = jnp.atleast_2d(batch_keys)
    means = jax.lax.map(f, xs=batch_keys)

    # Mean of batches is the mean of the total expectation
    return jnp.mean(means, axis=0)


def _hutchinson_batch(
    Q,
    /,
    *,
    tangents_shape,
    tangents_dtype,
    key,
    batch_size,
    generate_samples_fn=jax.random.rademacher,
):
    shape = (batch_size,) + tangents_shape
    samples = generate_samples_fn(key, shape=shape, dtype=tangents_dtype)
    return jnp.mean(jax.vmap(Q)(samples), axis=0)
