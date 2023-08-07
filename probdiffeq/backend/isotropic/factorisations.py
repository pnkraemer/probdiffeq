from probdiffeq.backend import _factorisations
from probdiffeq.backend.isotropic import (
    conditional,
    linearise,
    random,
    ssm_util,
    transform,
)


class IsotropicFactorisation(_factorisations.Factorisation):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def conditional(self):
        return conditional.ConditionalBackEnd()

    def linearise_ode(self):
        return linearise.LineariseODEBackEnd()

    def random(self):
        return random.RandomVariableBackEnd(ode_shape=self.ode_shape)

    def ssm_util(self):
        return ssm_util.SSMUtilBackEnd()

    def transform(self):
        return transform.TransformBackEnd()
