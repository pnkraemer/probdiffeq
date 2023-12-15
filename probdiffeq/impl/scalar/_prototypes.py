from probdiffeq.backend import numpy as np
from probdiffeq.impl import _prototypes
from probdiffeq.impl.scalar import _normal


class PrototypeBackend(_prototypes.PrototypeBackend):
    def qoi(self):
        return np.empty(())

    def observed(self):
        mean = np.empty(())
        cholesky = np.empty(())
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return np.empty(())

    def output_scale(self):
        return np.empty(())
