from probdiffeq.backend import functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
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
