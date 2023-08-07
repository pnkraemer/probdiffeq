import jax.numpy as jnp

from probdiffeq.backend import _random
from probdiffeq.backend.isotropic import _normal


class RandomVariableBackEnd(_random.RandomVariableBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def mahalanobis_norm(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        residual_white_matrix = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(residual_white_matrix), ())

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
