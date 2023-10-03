"""API for dense factorisations."""
from probdiffeq.impl import _impl
from probdiffeq.impl.dense import (
    _conditional,
    _hidden_model,
    _linearise,
    _prototypes,
    _ssm_util,
    _stats,
    _transform,
    _variable,
)


class Dense(_impl.FactorisedImpl):
    """Dense factorisation."""

    def __init__(self, ode_shape):
        """Construct a dense factorisation."""
        # TODO: add "order="F"" key
        self.ode_shape = ode_shape

    def linearise(self):
        return _linearise.LinearisationBackend(ode_shape=self.ode_shape)

    def hidden_model(self):
        return _hidden_model.HiddenModelBackend(ode_shape=self.ode_shape)

    def stats(self):
        return _stats.StatsBackend(ode_shape=self.ode_shape)

    def variable(self):
        return _variable.VariableBackend(ode_shape=self.ode_shape)

    def conditional(self):
        return _conditional.ConditionalBackend()

    def transform(self):
        return _transform.TransformBackend()

    def ssm_util(self):
        return _ssm_util.SSMUtilBackend(ode_shape=self.ode_shape)

    def prototypes(self):
        return _prototypes.PrototypeBackend(ode_shape=self.ode_shape)
