from probdiffeq.backend import abc, functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal


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


class DenseStats(StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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


class IsotropicStats(StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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


class BlockDiagStats(StatsBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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
