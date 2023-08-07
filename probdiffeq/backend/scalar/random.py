from typing import Any

import jax.numpy as jnp

from probdiffeq.backend import _random, containers
from probdiffeq.backend.scalar import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def qoi_like(self):
        mean = jnp.empty(())
        cholesky = jnp.empty(())
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return jnp.abs(res_white)

    def standard_deviation(self, rv):
        return jnp.abs(rv.cholesky)

    def qoi(self, rv):
        return rv.mean[0]

    def mean(self, rv):
        return rv.mean

    def rescale_cholesky(self, rv, factor):
        return _normal.Normal(rv.mean, factor * rv.cholesky)
