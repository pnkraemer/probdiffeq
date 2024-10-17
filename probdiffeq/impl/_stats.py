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
    def sample_shape(self, rv):
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
    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_from_sample(self, sample, /):
        raise NotImplementedError

    @abc.abstractmethod
    def update_mean(self, mean, x, /, num):
        raise NotImplementedError


class ScalarStats(StatsBackend):
    def mahalanobis_norm_relative(self, u, /, rv):
        res_white = (u - rv.mean) / rv.cholesky
        return np.abs(res_white) / np.sqrt(rv.mean.size)

    def logpdf(self, u, /, rv):
        dx = u - rv.mean
        w = linalg.solve_triangular(rv.cholesky.T, dx, trans="T")

        maha_term = linalg.vector_dot(w, w)

        diagonal = linalg.diagonal_along_axis(rv.cholesky, axis1=-1, axis2=-2)
        slogdet = np.sum(np.log(np.abs(diagonal)))
        logdet_term = 2.0 * slogdet
        return -0.5 * (logdet_term + maha_term + u.size * np.log(np.pi() * 2))

    def standard_deviation(self, rv):
        if rv.cholesky.ndim > 1:
            return functools.vmap(self.standard_deviation)(rv)

        return np.sqrt(linalg.vector_dot(rv.cholesky, rv.cholesky))

    def mean(self, rv):
        return rv.mean

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def rescale_cholesky(self, rv, factor):
        if np.ndim(factor) > 0:
            return functools.vmap(self.rescale_cholesky)(rv, factor)
        return _normal.Normal(rv.mean, factor * rv.cholesky)

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T

    def qoi(self, rv):
        return rv.mean[..., 0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        if i > rv.mean.shape[0]:
            raise ValueError

        m = rv.mean[i]
        c = rv.cholesky[[i], :]
        chol = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(np.reshape(m, ()), np.reshape(chol, ()))

    def qoi_from_sample(self, sample, /):
        return sample[0]

    def update_mean(self, mean, x, /, num):
        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)


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

    def sample_shape(self, rv):
        return rv.mean.shape

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T

    def qoi(self, rv):
        if np.ndim(rv.mean) > 1:
            return functools.vmap(self.qoi)(rv)
        return self.unravel(rv.mean)
        # mean_reshaped = np.reshape(rv.mean, (-1, *self.ode_shape), order="F")
        # return mean_reshaped[0]

    def marginal_nth_derivative(self, rv, i):
        if rv.mean.ndim > 1:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        m = self._select(rv.mean, i)
        c = functools.vmap(self._select, in_axes=(1, None), out_axes=1)(rv.cholesky, i)
        c = cholesky_util.triu_via_qr(c.T)
        return _normal.Normal(m, c.T)

    def qoi_from_sample(self, sample, /):
        sample_reshaped = np.reshape(sample, (-1, *self.ode_shape), order="F")
        return sample_reshaped[0]

    def _select(self, x, /, idx_or_slice):
        x_reshaped = np.reshape(x, (-1, *self.ode_shape), order="F")
        if isinstance(idx_or_slice, int) and idx_or_slice > x_reshaped.shape[0]:
            raise ValueError
        return x_reshaped[idx_or_slice]

    @staticmethod
    def _autobatch_linop(fun):
        def fun_(x):
            if np.ndim(x) > 1:
                return functools.vmap(fun_, in_axes=1, out_axes=1)(x)
            return fun(x)

        return fun_

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
        return np.sqrt(linalg.vector_dot(rv.cholesky, rv.cholesky))

    def sample_shape(self, rv):
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
        if rv.mean.ndim > 2:
            return functools.vmap(self.qoi)(rv)
        return self.unravel(rv.mean)

    def marginal_nth_derivative(self, rv, i):
        if np.ndim(rv.mean) > 2:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        if i > np.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[i, :]
        cholesky = cholesky_util.triu_via_qr(rv.cholesky[[i], :].T).T
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[0, :]

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

    def sample_shape(self, rv):
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
        return (mean, cov)

    def _cov_dense(self, cholesky):
        if cholesky.ndim > 2:
            return functools.vmap(self._cov_dense)(cholesky)
        return cholesky @ cholesky.T

    def qoi(self, rv):
        if rv.mean.ndim > 2:
            return functools.vmap(self.qoi)(rv)
        return self.unravel(rv.mean)

    def marginal_nth_derivative(self, rv, i):
        if np.ndim(rv.mean) > 2:
            return functools.vmap(self.marginal_nth_derivative, in_axes=(0, None))(
                rv, i
            )

        if i > np.shape(rv.mean)[0]:
            raise ValueError

        mean = rv.mean[:, i]
        cholesky = functools.vmap(cholesky_util.triu_via_qr)(
            (rv.cholesky[:, i, :])[..., None]
        )
        cholesky = np.transpose(cholesky, axes=(0, 2, 1))
        return _normal.Normal(mean, cholesky)

    def qoi_from_sample(self, sample, /):
        return sample[..., 0]

    def update_mean(self, mean, x, /, num):
        if np.ndim(mean) > 0:
            assert np.shape(mean) == np.shape(x)
            return functools.vmap(self.update_mean, in_axes=(0, 0, None))(mean, x, num)

        sum_updated = cholesky_util.sqrt_sum_square_scalar(np.sqrt(num) * mean, x)
        return sum_updated / np.sqrt(num + 1)
