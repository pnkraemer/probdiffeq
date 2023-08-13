"""Random variable implementation."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
    def mahalanobis_norm_relative(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return jnp.abs(res_white) / jnp.sqrt(rv.mean.size)

    def logpdf(self, u, /, rv):
        dx = u - rv.mean
        w = jax.scipy.linalg.solve_triangular(rv.cholesky.T, dx, trans="T")

        maha_term = jnp.dot(w, w)

        diagonal = jnp.diagonal(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * jnp.log(jnp.pi * 2))

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)

        return jnp.sqrt(jnp.dot(rv.cholesky, rv.cholesky))

    def mean(self, rv):
        return rv.mean

    def sample_shape(self, rv):
        return rv.mean.shape
