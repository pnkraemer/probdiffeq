import jax.numpy as jnp

from probdiffeq.impl import _random
from probdiffeq.impl.blockdiag import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm(self, u, /, rv):
        return (rv.mean - u) / rv.cholesky  # return array of norms! See calibration

    def logpdf(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        x1 = jnp.square(residual_white)
        x2 = u.size * 2.0 * jnp.log(jnp.abs(rv.cholesky))
        x3 = u.size * jnp.log(jnp.pi * 2)
        return jnp.sum(-0.5 * (x1 + x2 + x3))

    def mean(self, rv):
        return rv.mean

    def qoi_like(self):
        mean_and_cholesky = jnp.empty(self.ode_shape)
        return _normal.Normal(mean_and_cholesky, mean_and_cholesky)

    def qoi(self, rv):
        return rv.mean[..., 0]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        return jnp.abs(
            rv.cholesky
        )  # todo: this is only true of rv is an "observed" Rv...
