import jax.numpy as jnp

from probdiffeq.statespace import _random
from probdiffeq.statespace.isotropic import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        residual_white_matrix = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(residual_white_matrix), ())

    def logpdf(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        x1 = jnp.dot(residual_white, residual_white)
        x2 = u.size * 2.0 * jnp.log(jnp.abs(rv.cholesky))
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mean(self, rv):
        return rv.mean

    def qoi_like(self):
        mean = jnp.empty(self.ode_shape)
        cholesky = jnp.empty(())
        return _normal.Normal(mean, cholesky)

    def qoi(self, rv):
        return rv.mean[0, :]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        return rv.cholesky  # todo: this is only true of rv is an "observed" Rv...
