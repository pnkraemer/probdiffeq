"""Low-level tests for Hutchinson's trace estimation (and related routines)."""


import jax
import jax.numpy as jnp
import pytest_cases

from probdiffeq.implementations import _hutch as hutch


@pytest_cases.fixture(name="fn")
def fixture_fn():
    def f(x):
        return jnp.sin(jnp.flip(jnp.cos(x)) + 1.0) * jnp.sin(x) + 1.0

    return f


@pytest_cases.fixture(name="rng_key")
def fixture_rng_key():
    return jax.random.PRNGKey(seed=1)


@pytest_cases.parametrize("num_batches", [1, 1_000])
@pytest_cases.parametrize("batch_size", [1, 1_000])
@pytest_cases.parametrize("dim", [1, 100])
@pytest_cases.parametrize(
    "generate_samples_fn", [jax.random.normal, jax.random.rademacher]
)
def test_trace(fn, rng_key, num_batches, batch_size, dim, generate_samples_fn):
    # Linearise function
    x0 = jax.random.uniform(rng_key, shape=(dim,))
    _, jvp = jax.linearize(fn, x0)
    J = jax.jacfwd(fn)(x0)

    # Sequential batches
    keys = jax.random.split(rng_key, num=num_batches)

    # Estimate the trace
    estimate = hutch.trace(
        matvec_fn=jvp,
        tangents_shape=jnp.shape(x0),
        tangents_dtype=jnp.dtype(x0),
        batch_keys=keys,
        batch_size=batch_size,
        generate_samples_fn=generate_samples_fn,
    )
    truth = jnp.trace(J)
    assert jnp.allclose(estimate, truth, atol=1e-1, rtol=1e-1)
