from probdiffeq.backend import abc, functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Callable
from probdiffeq.impl import _normal
from probdiffeq.util import cholesky_util


class StatsBackend(abc.ABC):
    @abc.abstractmethod
    def mahalanobis_norm_relative(self, u, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def logpdf(self, u, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_deviation(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def hidden_shape(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, unit_sample, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def to_multivariate_normal(self, u, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_from_sample(self, sample, /):
        raise NotImplementedError

    @abc.abstractmethod
    def update_mean(self, mean, x, /, num):
        raise NotImplementedError


class DenseStats(StatsBackend):
    def __init__(self, ode_shape: tuple, unravel: Callable):
        self.ode_shape = ode_shape
        self.unravel = unravel

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = linalg.solve_triangular(
            rv.cholesky.T, u - rv.mean, lower=False, trans="T"
        )
        mahalanobis = linalg.qr_r(residual_white[:, None])
        return np.reshape(np.abs(mahalanobis) / np.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # The cholesky factor is triangular, so we compute a cheap slogdet.
        diagonal = linalg.diagonal_along_axis(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))

        dx = u - rv.mean
        residual_white = linalg.solve_triangular(rv.cholesky.T, dx, trans="T")
        x1 = linalg.vector_dot(residual_white, residual_white)
        x2 = 2.0 * slogdet
        x3 = u.size * np.log(np.pi() * 2)
        return -0.5 * (x1 + x2 + x3)

    def mean(self, rv):
        return rv.mean

    def standard_deviation(self, rv):
        if rv.mean.ndim > 1:
            return functools.vmap(self.standard_deviation)(rv)

        diag = np.einsum("ij,ij->i", rv.cholesky, rv.cholesky)
        return np.sqrt(diag)

    def hidden_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T

    def qoi(self, rv):
        return self.qoi_from_sample(rv.mean)

    def qoi_from_sample(self, sample, /):
        if np.ndim(sample) > 1:
            return functools.vmap(self.qoi_from_sample)(sample)
        return self.unravel(sample)

    def update_mean(self, mean, x, /, num):
        nominator = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        denominator = np.sqrt(num + 1)
        return nominator / denominator


class IsotropicStats(StatsBackend):
    def __init__(self, ode_shape, unravel):
        self.ode_shape = ode_shape
        self.unravel = unravel

    def mahalanobis_norm_relative(self, u, /, rv):
        residual_white = (rv.mean - u) / rv.cholesky
        residual_white_matrix = linalg.qr_r(residual_white.T)
        return np.reshape(np.abs(residual_white_matrix) / np.sqrt(rv.mean.size), ())

    def logpdf(self, u, /, rv):
        # if the gain is qoi-to-hidden, the data is a (d,) array.
        # this is problematic for the isotropic model unless we explicitly broadcast.
        if np.ndim(u) == 1:
            u = u[None, :]

        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = linalg.solve_triangular(r.cholesky.T, dx, trans="T")

            maha_term = linalg.vector_dot(w, w)

            diagonal = linalg.diagonal_along_axis(r.cholesky, axis1=-1, axis2=-2)
            slogdet = np.sum(np.log(np.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * np.log(np.pi() * 2))

        # Batch in the "mean" dimension and sum the results.
        rv_batch = _normal.Normal(1, None)
        return np.sum(functools.vmap(logpdf_scalar, in_axes=(1, rv_batch))(u, rv))

    def mean(self, rv):
        return rv.mean

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return functools.vmap(self.standard_deviation)(rv)
        std = np.sqrt(linalg.vector_dot(rv.cholesky, rv.cholesky))
        return std[..., None] @ np.ones((1, rv.mean.shape[-1]))

    def hidden_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        eye_d = np.eye(*self.ode_shape)
        cov = rv.cholesky @ rv.cholesky.T
        cov = np.kron(eye_d, cov)
        mean = rv.mean.reshape((-1,), order="F")
        return (mean, cov)

    def qoi(self, rv):
        return self.qoi_from_sample(rv.mean)

    def qoi_from_sample(self, sample, /):
        if np.ndim(sample) > 2:
            return functools.vmap(self.qoi_from_sample)(sample)
        return self.unravel(sample)

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)


class BlockDiagStats(StatsBackend):
    def __init__(self, ode_shape, unravel):
        self.ode_shape = ode_shape
        self.unravel = unravel

    def mahalanobis_norm_relative(self, u, /, rv):
        # assumes rv.chol = (d,1,1)
        # return array of norms! See calibration
        mean = np.reshape(rv.mean, self.ode_shape)
        cholesky = np.reshape(rv.cholesky, self.ode_shape)
        return (mean - u) / cholesky / np.sqrt(mean.size)

    def logpdf(self, u, /, rv):
        def logpdf_scalar(x, r):
            dx = x - r.mean
            w = linalg.solve_triangular(r.cholesky.T, dx, trans="T")

            maha_term = linalg.vector_dot(w, w)

            diagonal = linalg.diagonal_along_axis(r.cholesky, axis1=-1, axis2=-2)
            slogdet = np.sum(np.log(np.abs(diagonal)))
            logdet_term = 2.0 * slogdet
            return -0.5 * (logdet_term + maha_term + x.size * np.log(np.pi() * 2))

        return np.sum(functools.vmap(logpdf_scalar)(u, rv))

    def mean(self, rv):
        return rv.mean

    def hidden_shape(self, rv):
        return rv.mean.shape

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return functools.vmap(self.standard_deviation)(rv)

        return np.sqrt(linalg.vector_dot(rv.cholesky, rv.cholesky))

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + (rv.cholesky @ unit_sample[..., None])[..., 0]

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        mean = np.reshape(rv.mean.T, (-1,), order="F")
        cov = np.block_diag(self._cov_dense(rv.cholesky))
        return mean, cov

    def _cov_dense(self, cholesky):
        if cholesky.ndim > 2:
            return functools.vmap(self._cov_dense)(cholesky)
        return cholesky @ cholesky.T

    def qoi(self, rv):
        return self.qoi_from_sample(rv.mean)

    def qoi_from_sample(self, sample, /):
        if np.ndim(sample) > 2:
            return functools.vmap(self.qoi_from_sample)(sample)
        return self.unravel(sample)

    def update_mean(self, mean, x, /, num):
        if np.ndim(mean) > 0:
            assert np.shape(mean) == np.shape(x)
            return functools.vmap(self.update_mean, in_axes=(0, 0, None))(mean, x, num)

        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)
