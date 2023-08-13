"""Scalar factorisation."""
from probdiffeq.impl import _impl
from probdiffeq.impl.scalar import (
    _conditional,
    _hidden_model,
    _linearise,
    _prototypes,
    _ssm_util,
    _stats,
    _transform,
    _variable,
)


class Scalar(_impl.FactorisedImpl):
    """Scalar factorisation."""

    def linearise(self):
        return _linearise.LinearisationBackend()

    def conditional(self):
        return _conditional.ConditionalBackend()

    def transform(self):
        return _transform.TransformBackend()

    def ssm_util(self):
        return _ssm_util.SSMUtilBackend()

    def prototypes(self):
        return _prototypes.PrototypeBackend()

    def hidden_model(self):
        return _hidden_model.HiddenModelBackend()

    def stats(self):
        return _stats.StatsBackend()

    def variable(self):
        return _variable.VariableBackend()
