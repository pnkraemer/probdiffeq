import jax.numpy as jnp

from probdiffeq.backend import _random
from probdiffeq.backend.blockdiag import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm(self, u, /, rv):
        return (rv.mean - u) / rv.cholesky  # return array of norms! See calibration

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
