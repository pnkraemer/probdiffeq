"""Random variable implementation."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _random, _sqrt_util
from probdiffeq.impl.dense import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = jax.scipy.linalg.solve_triangular(
            rv.cholesky.T, u - rv.mean, lower=False, trans="T"
        )
        mahalanobis = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(mahalanobis) / jnp.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # The cholesky factor is triangular, so we compute a cheap slogdet.
        # todo: cache those?
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

    def qoi(self, rv):
        if jnp.ndim(rv.mean) > 1:
            return jax.vmap(self.qoi)(rv)
        mean_reshaped = jnp.reshape(rv.mean, (-1, *self.ode_shape), order="F")
        return mean_reshaped[0]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        if rv.mean.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)

        diag = jnp.einsum("ij,ij->i", rv.cholesky, rv.cholesky)
        return jnp.sqrt(diag)

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if jnp.ndim(rv.cholesky) > 2:
            return jax.vmap(self.cov_dense)(rv)
        return rv.cholesky @ rv.cholesky.T

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        m = self._select(rv.mean, i)
        c = jax.vmap(self._select, in_axes=(1, None), out_axes=1)(rv.cholesky, i)
        c = _sqrt_util.triu_via_qr(c.T)
        return _normal.Normal(m, c.T)

    def _select(self, x, /, i):
        x_reshaped = jnp.reshape(x, (-1, *self.ode_shape), order="F")
        if i > x_reshaped.shape[0]:
            raise ValueError
        return x_reshaped[i]

    def qoi_from_sample(self, sample, /):
        sample_reshaped = jnp.reshape(sample, (-1, *self.ode_shape), order="F")
        return sample_reshaped[0]

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, u, rv):
        return u, (rv.mean, rv.cholesky @ rv.cholesky.T)
