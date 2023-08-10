"""Random variable implementation."""
import jax
import jax.numpy as jnp

from probdiffeq.impl import _sqrt_util  # todo: get sqrt-util into "impl" package...
from probdiffeq.impl import _random
from probdiffeq.impl.isotropic import _normal


class RandomVariableBackend(_random.RandomVariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def variable(self, mean, cholesky):
        return _normal.Normal(mean, cholesky)

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        residual_white_matrix = jnp.linalg.qr(residual_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(residual_white_matrix) / jnp.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if jnp.ndim(u) == 1:
            u = u[None, :]

        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = jax.scipy.linalg.solve_triangular(r.cholesky, dx, lower=True, trans="T")

            maha_term = jnp.dot(w, w)

            diagonal = jnp.diagonal(r.cholesky, axis1=-1, axis2=-2)
            slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * jnp.log(jnp.pi * 2))

        # Batch in the "mean" dimension and sum the results.
        rv_batch = _normal.Normal(1, None)
        return jnp.sum(jax.vmap(logpdf_scalar, in_axes=(1, rv_batch))(u, rv))

    def mean(self, rv):
        return rv.mean

    def qoi(self, rv):
        return rv.mean[..., 0, :]

    def cholesky(self, rv):
        return rv.cholesky

    def cov_dense(self, rv):
        if rv.cholesky.ndim > 2:
            return jax.vmap(self.cov_dense)(rv)
        return rv.cholesky @ rv.cholesky.T

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 2:
            return jax.vmap(self.standard_deviation)(rv)

        diag = jnp.einsum("ij,ij->i", rv.cholesky, rv.cholesky)
        return jnp.sqrt(diag)

    def marginal_nth_derivative(self, rv, i):
        if jnp.ndim(rv.mean) > 2:
            return jax.vmap(self.marginal_nth_derivative, in_axes=(0, None))(rv, i)

        if i > jnp.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[i, :]
        cholesky = _sqrt_util.triu_via_qr((rv.cholesky[i, :])[:, None].T).T
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[0, :]

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, u, rv):
        eye_d = jnp.eye(*self.ode_shape)
        cov = rv.cholesky @ rv.cholesky.T
        cov = jnp.kron(eye_d, cov)
        mean = rv.mean.reshape((-1,), order="F")
        u = u.reshape((-1,), order="F")
        return u, (mean, cov)
