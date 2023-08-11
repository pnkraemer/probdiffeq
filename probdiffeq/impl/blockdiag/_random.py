"""Random variable implementations."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _random, sqrt_util
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
        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = jax.scipy.linalg.solve_triangular(r.cholesky, dx, lower=True, trans="T")

            maha_term = jnp.dot(w, w)

            diagonal = jnp.diagonal(r.cholesky, axis1=-1, axis2=-2)
            slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * jnp.log(jnp.pi * 2))

        return jnp.sum(jax.vmap(logpdf_scalar)(u, rv))

    def mean(self, rv):
        return rv.mean

    def qoi(self, rv):
        return rv.mean[..., 0]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return jax.vmap(self.standard_deviation)(rv)

        return jnp.sqrt(jnp.dot(rv.cholesky, rv.cholesky))

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if rv.cholesky.ndim > 3:
            return jax.vmap(self.cov_dense)(rv)
        cholesky_T = jnp.transpose(rv.cholesky, axes=(0, 2, 1))
        return jnp.einsum("ijk,ikl->ijl", rv.cholesky, cholesky_T)

    def marginal_nth_derivative(self, rv, i):
        if jnp.ndim(rv.mean) > 2:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        if i > jnp.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[:, i]
        cholesky = jax.vmap(sqrt_util.triu_via_qr)((rv.cholesky[:, i, :])[..., None])
        cholesky = jnp.transpose(cholesky, axes=(0, 2, 1))
        return _normal.Normal(mean, cholesky)

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + (rv.cholesky @ unit_sample[..., None])[..., 0]

    def qoi_from_sample(self, sample, /):
        return sample[..., 0]

    def to_multivariate_normal(self, u, rv):
        mean = jnp.reshape(rv.mean.T, (-1,), order="F")
        u = jnp.reshape(u.T, (-1,), order="F")
        cov = jax.scipy.linalg.block_diag(*self.cov_dense(rv))
        return u, (mean, cov)
