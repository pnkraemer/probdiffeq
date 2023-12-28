"""Stats."""

import jax.scipy.stats


def multivariate_normal_logpdf(x, /, mean, cov):
    return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)
