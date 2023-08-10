"""API for dense factorisations."""
from probdiffeq.impl import _factorisations
from probdiffeq.impl.dense import (
    _conditional,
    _linearise,
    _random,
    _ssm_util,
    _transform,
)


class DenseFactorisation(_factorisations.Factorisation):
    """Dense factorisation."""

    def __init__(self, ode_shape):
        """Construct a dense factorisation."""
        # todo: add "order="F"" key
        self.ode_shape = ode_shape

    def linearise(self):
        return _linearise.LinearisationBackend(ode_shape=self.ode_shape)

    def random(self):
        return _random.RandomVariableBackend(ode_shape=self.ode_shape)

    def conditional(self):
        return _conditional.ConditionalBackend()

    def transform(self):
        return _transform.TransformBackend()

    def ssm_util(self):
        return _ssm_util.SSMUtilBackend(ode_shape=self.ode_shape)
