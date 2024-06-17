from probdiffeq.backend import abc, functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal


class VariableBackend(abc.ABC):
    @abc.abstractmethod
    def to_multivariate_normal(self, u, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError


class ScalarVariable(VariableBackend):
    def rescale_cholesky(self, rv, factor):
        if np.ndim(factor) > 0:
            return functools.vmap(self.rescale_cholesky)(rv, factor)
        return _normal.Normal(rv.mean, factor * rv.cholesky)

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T


class DenseVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T


class IsotropicVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def to_multivariate_normal(self, rv):
        eye_d = np.eye(*self.ode_shape)
        cov = rv.cholesky @ rv.cholesky.T
        cov = np.kron(eye_d, cov)
        mean = rv.mean.reshape((-1,), order="F")
        return (mean, cov)


class BlockDiagVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

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
