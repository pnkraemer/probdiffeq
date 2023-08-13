import jax
import jax.numpy as jnp

from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def mahalanobis_norm_relative(self, u, /, rv):
        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        mean = jnp.reshape(rv.mean, self.ode_shape)
        cholesky = jnp.reshape(rv.cholesky, self.ode_shape)
        return (mean - u) / cholesky / jnp.sqrt(mean.size)

    def logpdf(self, u, /, rv):
        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = jax.scipy.linalg.solve_triangular(r.cholesky.T, dx, trans="T")

            maha_term = jnp.dot(w, w)

            diagonal = jnp.diagonal(r.cholesky, axis1=-1, axis2=-2)
            slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * jnp.log(jnp.pi * 2))

        return jnp.sum(jax.vmap(logpdf_scalar)(u, rv))

    def mean(self, rv):
        return rv.mean

    def sample_shape(self, rv):
        return rv.mean.shape

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)

        return jnp.sqrt(jnp.dot(rv.cholesky, rv.cholesky))
