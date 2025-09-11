"""(Pseudo)random number generation."""

import jax.random


def prng_key(*, seed):
    return jax.random.PRNGKey(seed=seed)


def split(key, num):
    return jax.random.split(key, num=num)


def normal(key, /, shape):
    return jax.random.normal(key, shape=shape)


def rademacher(key, /, shape, dtype):
    return jax.random.rademacher(key, shape=shape, dtype=dtype)
