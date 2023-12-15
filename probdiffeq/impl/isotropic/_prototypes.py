from probdiffeq.backend import numpy as np
from probdiffeq.impl import _prototypes
from probdiffeq.impl.isotropic import _normal


class PrototypeBackend(_prototypes.PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self):
        return np.empty(self.ode_shape)

    def observed(self):
        mean = np.empty((1, *self.ode_shape))
        cholesky = np.empty(())
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return np.empty(())

    def output_scale(self):
        return np.empty(())
