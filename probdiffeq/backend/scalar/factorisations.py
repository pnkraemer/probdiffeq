from probdiffeq.backend import _factorisations
from probdiffeq.backend.scalar import (
    conditional,
    linearise,
    random,
    ssm_util,
    transform,
)


class ScalarFactorisation(_factorisations.Factorisation):
    def linearise_ode(self):
        return linearise.LineariseODEBackEnd()

    def random(self):
        return random.RandomVariableBackEnd()

    def conditional(self):
        return conditional.ConditionalBackEnd()

    def transform(self):
        return transform.TransformBackEnd()

    def ssm_util(self):
        return ssm_util.SSMUtilBackEnd()
