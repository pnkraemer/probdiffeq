"""Scalar factorisation."""
from probdiffeq.impl import _factorisations
from probdiffeq.impl.scalar import (
    _conditional,
    _linearise,
    _random,
    _ssm_util,
    _transform,
)


class ScalarFactorisation(_factorisations.Factorisation):
    """Scalar factorisation."""

    def linearise(self):
        return _linearise.LinearisationBackend()

    def random(self):
        return _random.RandomVariableBackend()

    def conditional(self):
        return _conditional.ConditionalBackend()

    def transform(self):
        return _transform.TransformBackend()

    def ssm_util(self):
        return _ssm_util.SSMUtilBackend()
