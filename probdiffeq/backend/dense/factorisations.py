from probdiffeq.backend import _factorisations
from probdiffeq.backend.dense import cond, linearise, random, ssm_util


class DenseFactorisation(_factorisations.Factorisation):
    def __init__(self, ode_shape):
        # todo: add "order="F"" key
        self.ode_shape = ode_shape

    def linearise_ode(self):
        return linearise.LineariseODEBackEnd(ode_shape=self.ode_shape)

    def random(self):
        return random.RandomVariableBackEnd(ode_shape=self.ode_shape)

    def cond(self):
        return cond.ConditionalBackEnd(ode_shape=self.ode_shape)

    def ssm_util(self):
        return ssm_util.SSMUtilBackEnd(ode_shape=self.ode_shape)
