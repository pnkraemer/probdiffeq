from probdiffeq.backend import numpy as np
from probdiffeq.impl import _variable
from probdiffeq.impl.isotropic import _normal


class VariableBackend(_variable.VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def rescale_cholesky(self, rv, factor, /):
        cholesky = factor[..., None, None] * rv.cholesky
        return _normal.Normal(rv.mean, cholesky)

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, rv):
        eye_d = np.eye(*self.ode_shape)
        cov = rv.cholesky @ rv.cholesky.T
        cov = np.kron(eye_d, cov)
        mean = rv.mean.reshape((-1,), order="F")
        return (mean, cov)
