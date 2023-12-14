"""(Pseudo)random number generation."""

import jax.random


def prng_key(*, seed):
    return jax.random.PRNGKey(seed=seed)


def normal(key, /, shape):
    return jax.random.normal(key, shape=shape)
