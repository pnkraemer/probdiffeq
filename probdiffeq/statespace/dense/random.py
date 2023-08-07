from typing import Any

import jax
import jax.numpy as jnp

from probdiffeq.backend import containers
from probdiffeq.statespace import _random
from probdiffeq.statespace.dense import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm(self, u, /, rv):
        residual_white = jax.scipy.linalg.solve_triangular(
            rv.cholesky.T, u - rv.mean, lower=False, trans="T"
        )
        mahalanobis = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(mahalanobis), ())

    def logpdf(self, u, /, rv):
        # The cholesky factor is triangular, so we compute a cheap slogdet.
        # todo: cache those?
        diagonal = jnp.diagonal(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        residual_white = jax.scipy.linalg.solve_triangular(
            rv.cholesky.T, u - rv.mean, lower=False, trans="T"
        )
        x1 = jnp.dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mean(self, rv):
        return rv.mean

    def qoi(self, rv):
        mean = rv.mean
        mean_reshaped = jnp.reshape(mean, (-1,) + self.ode_shape, order="F")
        return mean_reshaped[0]

    def qoi_like(self):
        mean = jnp.empty(self.ode_shape)
        cholesky = jnp.empty(self.ode_shape + self.ode_shape)
        return _normal.Normal(mean, cholesky)

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        def std(x):
            std_mat = jnp.linalg.qr(x[..., None], mode="r")
            return jnp.reshape(std_mat, ())

        return jax.vmap(std)(rv.cholesky)
