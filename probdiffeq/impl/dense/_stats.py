from probdiffeq.backend import functools, linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _stats


class StatsBackend(_stats.StatsBackend):
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
