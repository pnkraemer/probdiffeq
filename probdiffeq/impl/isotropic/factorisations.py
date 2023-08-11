"""Isotropic factorisation."""
from probdiffeq.impl import _factorisations
from probdiffeq.impl.isotropic import (
    _conditional,
    _linearise,
    _prototypes,
    _random,
    _ssm_util,
    _transform,
)


class IsotropicFactorisation(_factorisations.Factorisation):
    """Isotropic factorisation."""

    def __init__(self, ode_shape):
        """Construct an isotropic factorisation."""
        self.ode_shape = ode_shape

    def conditional(self):
        return _conditional.ConditionalBackend()

    def linearise(self):
        return _linearise.LinearisationBackend()

    def random(self):
        return _random.RandomVariableBackend(ode_shape=self.ode_shape)

    def ssm_util(self):
        return _ssm_util.SSMUtilBackend(ode_shape=self.ode_shape)

    def transform(self):
        return _transform.TransformBackend()

    def prototypes(self):
        return _prototypes.PrototypeBackend(ode_shape=self.ode_shape)
