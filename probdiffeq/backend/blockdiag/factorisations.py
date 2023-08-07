from probdiffeq.backend import _factorisations
from probdiffeq.backend.blockdiag import (
    conditional,
    linearise,
    random,
    ssm_util,
    transform,
)


class BlockDiagFactorisation(_factorisations.Factorisation):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def conditional(self):
        raise NotImplementedError

    def linearise(self):
        return linearise.LinearisationBackend()

    def random(self):
        raise NotImplementedError

    def ssm_util(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError
