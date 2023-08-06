from typing import Any

import jax.numpy as jnp

from probdiffeq.backend import _random, containers


class Normal(containers.NamedTuple):
    mean: Any
    cholesky: Any


class RandomVariableBackEnd(_random.RandomVariableBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def mahalanobis_norm(self, u, /, rv):
        raise NotImplementedError

    def mean(self, rv):
        return rv.mean

    def qoi(self, rv):
        mean = rv.mean
        mean_reshaped = jnp.reshape(mean, (-1,) + self.ode_shape, order="F")
        return mean_reshaped[0]

    def qoi_like(self):
        mean = jnp.empty(self.ode_shape)
        cholesky = jnp.empty(self.ode_shape + self.ode_shape)
        return Normal(mean, cholesky)

    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError

    def standard_deviation(self, rv):
        raise NotImplementedError
