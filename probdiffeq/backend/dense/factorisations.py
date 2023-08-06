from probdiffeq.backend import _factorisations
from probdiffeq.backend.dense import cond, linearise, random, ssm_util


class DenseFactorisation(_factorisations.Factorisation):
    def __init__(self, ode_shape, order="F"):
        self.ode_shape = ode_shape
        self.order = order

    def linearise_ode(self):
        return linearise.LineariseODEBackEnd(ode_shape=ode_shape, order=order)

    def random(self):
        return random.RandomVariableBackEnd(ode_shape=ode_shape, order=order)

    def cond(self):
        return cond.ConditionalBackEnd(ode_shape=ode_shape, order=order)

    def ssm_util(self):
        return ssm_util.SSMUtilBackEnd(ode_shape=ode_shape, order=order)
