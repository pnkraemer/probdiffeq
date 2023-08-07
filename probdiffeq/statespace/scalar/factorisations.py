from probdiffeq.statespace import _factorisations
from probdiffeq.statespace.scalar import (
    conditional,
    linearise,
    random,
    ssm_util,
    transform,
)


class ScalarFactorisation(_factorisations.Factorisation):
    def linearise(self):
        return linearise.LinearisationBackend()

    def random(self):
        return random.RandomVariableBackend()

    def conditional(self):
        return conditional.ConditionalBackend()

    def transform(self):
        return transform.TransformBackend()

    def ssm_util(self):
        return ssm_util.SSMUtilBackend()
