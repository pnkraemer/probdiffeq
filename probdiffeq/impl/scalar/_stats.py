"""Random variable implementation."""
import jax.numpy as jnp

from probdiffeq.backend import functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
    def mahalanobis_norm_relative(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return np.abs(res_white) / np.sqrt(rv.mean.size)

    def logpdf(self, u, /, rv):
        dx = u - rv.mean
        w = linalg.solve_triangular(rv.cholesky.T, dx, trans="T")

        maha_term = jnp.dot(w, w)

        diagonal = jnp.diagonal(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * jnp.log(jnp.pi * 2))

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return functools.vmap(self.standard_deviation)(rv)

        return np.sqrt(jnp.dot(rv.cholesky, rv.cholesky))

    def mean(self, rv):
        return rv.mean

    def sample_shape(self, rv):
        return rv.mean.shape
