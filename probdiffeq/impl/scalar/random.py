"""Random variable implementation."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _random
from probdiffeq.impl.scalar import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm_relative(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return jnp.abs(res_white) / jnp.sqrt(rv.mean.size)

    def logpdf(self, u, /, rv):
        x1 = 2.0 * jnp.log(jnp.abs(rv.cholesky))  # logdet
        residual_white = (u - rv.mean) / rv.cholesky
        x2 = jnp.square(residual_white)
        x3 = jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def standard_deviation(self, rv):
        return jnp.abs(rv.cholesky)

    def qoi(self, rv):
        return rv.mean[0]

    def mean(self, rv):
        return rv.mean

    def rescale_cholesky(self, rv, factor):
        return _normal.Normal(rv.mean, factor * rv.cholesky)

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if rv.mean.ndim > 1:
            return jax.vmap(self.cov_dense)(rv)
        return rv.cholesky @ rv.cholesky.T

    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    def qoi_from_sample(self, sample, /):
        return sample[0]

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample
