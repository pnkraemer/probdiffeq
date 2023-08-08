"""Random variable implementations."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _random
from probdiffeq.impl.blockdiag import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm_relative(self, u, /, rv):
        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        mean = jnp.reshape(rv.mean, self.ode_shape)
        cholesky = jnp.reshape(rv.cholesky, self.ode_shape)
        return (mean - u) / cholesky / jnp.sqrt(mean.size)

    def logpdf(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        x1 = jnp.square(residual_white)
        x2 = u.size * 2.0 * jnp.log(jnp.abs(rv.cholesky))
        x3 = u.size * jnp.log(jnp.pi * 2)
        return jnp.sum(-0.5 * (x1 + x2 + x3))

    def mean(self, rv):
        return rv.mean

    def qoi_like(self):
        mean = jnp.empty(self.ode_shape + (1,))
        cholesky = jnp.empty(self.ode_shape + (1, 1))
        return _normal.Normal(mean, cholesky)

    def qoi(self, rv):
        return rv.mean[..., 0]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        # assumes rv.chol = (d,1,1)
        return jnp.abs(jnp.reshape(rv.cholesky, (-1,)))

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if rv.cholesky.ndim > 3:
            return jax.vmap(self.cov_dense)(rv)
        cholesky_T = jnp.transpose(rv.cholesky, axes=(0, 2, 1))
        return jnp.einsum("ijk,ikl->ijl", rv.cholesky, cholesky_T)

    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + (rv.cholesky @ unit_sample[..., None])[..., 0]

    def qoi_from_sample(self, sample, /):
        return sample[..., 0]
