from probdiffeq.statespace.backend import factorisations
from probdiffeq.statespace.backend.scalar import cond, linearise, random, ssm_util


class ScalarFactorisation(factorisations.Factorisation):
    def linearise_ode(self):
        return linearise.LineariseODEBackEnd()

    def random(self):
        return random.RandomVariableBackEnd()

    def cond(self):
        return cond.ConditionalBackEnd()

    def ssm_util(self):
        return ssm_util.SSMUtilBackEnd()
