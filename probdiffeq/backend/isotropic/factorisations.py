from probdiffeq.backend import _factorisations
from probdiffeq.backend.isotropic import conditional, linearise, random, ssm_util


class IsotropicFactorisation(_factorisations.Factorisation):
    def conditional(self):
        raise NotImplementedError

    def linearise_ode(self):
        return linearise.LineariseODEBackEnd()

    def random(self):
        raise NotImplementedError

    def ssm_util(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError
