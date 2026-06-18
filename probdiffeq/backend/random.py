"""(Pseudo)random number generation."""

import jax.random
import jax.scipy.stats


def prng_key(*, seed):
    return jax.random.PRNGKey(seed=seed)


def split(key, num):
    return jax.random.split(key, num=num)


def normal(key, /, shape, dtype=None):
    return jax.random.normal(key, shape=shape, dtype=dtype)


def rademacher(key, /, shape, dtype):
    return jax.random.rademacher(key, shape=shape, dtype=dtype)


def logpdf_multivariate_normal(x, /, mean, cov):
    return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)
