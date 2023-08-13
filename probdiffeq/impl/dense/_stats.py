import jax
import jax.numpy as jnp

from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = jax.scipy.linalg.solve_triangular(
            rv.cholesky.T, u - rv.mean, lower=False, trans="T"
        )
        mahalanobis = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(mahalanobis) / jnp.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # The cholesky factor is triangular, so we compute a cheap slogdet.
        # TODO: cache those?
        diagonal = jnp.diagonal(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        residual_white = jax.scipy.linalg.solve_triangular(
            rv.cholesky, u - rv.mean, lower=True, trans="T"
        )
        x1 = jnp.dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mean(self, rv):
        return rv.mean

    def standard_deviation(self, rv):
        if rv.mean.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)

        diag = jnp.einsum("ij,ij->i", rv.cholesky, rv.cholesky)
        return jnp.sqrt(diag)

    def cholesky(self, rv):
        return rv.cholesky

    def sample_shape(self, rv):
        return rv.mean.shape
