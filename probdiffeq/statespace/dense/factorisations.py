from probdiffeq.statespace import _factorisations
from probdiffeq.statespace.dense import (
    conditional,
    linearise,
    random,
    ssm_util,
    transform,
)


class DenseFactorisation(_factorisations.Factorisation):
    def __init__(self, ode_shape):
        # todo: add "order="F"" key
        self.ode_shape = ode_shape

    def linearise(self):
        return linearise.LinearisationBackend(ode_shape=self.ode_shape)

    def random(self):
        return random.RandomVariableBackend(ode_shape=self.ode_shape)

    def conditional(self):
        return conditional.ConditionalBackend()

    def transform(self):
        return transform.TransformBackend()

    def ssm_util(self):
        return ssm_util.SSMUtilBackend(ode_shape=self.ode_shape)
