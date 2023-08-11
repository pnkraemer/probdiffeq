"""Random variable implementation."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _random
from probdiffeq.impl.scalar import _normal
from probdiffeq.impl.util import cholesky_util


class RandomVariableBackend(_random.RandomVariableBackend):
    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm_relative(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return jnp.abs(res_white) / jnp.sqrt(rv.mean.size)

    def logpdf(self, u, /, rv):
        dx = u - rv.mean
        w = jax.scipy.linalg.solve_triangular(rv.cholesky, dx, lower=True, trans="T")

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

    def rescale_cholesky(self, rv, factor):
        if jnp.ndim(factor) > 0:
            return jax.vmap(self.rescale_cholesky)(rv, factor)
        return _normal.Normal(rv.mean, factor * rv.cholesky)

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if rv.mean.ndim > 1:
            return jax.vmap(self.cov_dense)(rv)
        return rv.cholesky @ rv.cholesky.T

    def qoi(self, rv):
        return rv.mean[..., 0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        if i > rv.mean.shape[0]:
            raise ValueError

        m = rv.mean[i]
        c = rv.cholesky[[i], :]
        chol = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(jnp.reshape(m, ()), jnp.reshape(chol, ()))

    def qoi_from_sample(self, sample, /):
        return sample[0]

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, u, rv):
        return u, (rv.mean, rv.cholesky @ rv.cholesky.T)
