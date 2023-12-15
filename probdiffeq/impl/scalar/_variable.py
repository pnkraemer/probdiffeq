"""Random variable implementation."""

from probdiffeq.backend import functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _variable
from probdiffeq.impl.scalar import _normal


class VariableBackend(_variable.VariableBackend):
    def rescale_cholesky(self, rv, factor):
        if np.ndim(factor) > 0:
            return functools.vmap(self.rescale_cholesky)(rv, factor)
        return _normal.Normal(rv.mean, factor * rv.cholesky)

    def transform_unit_sample(self, unit_sample, /, rv):
        return rv.mean + rv.cholesky @ unit_sample

    def to_multivariate_normal(self, rv):
        return rv.mean, rv.cholesky @ rv.cholesky.T
