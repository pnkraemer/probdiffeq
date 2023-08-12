"""State-space model impl."""

from probdiffeq.impl import (
    _conditional,
    _factorisations,
    _hidden_model,
    _linearise,
    _prototypes,
    _ssm_util,
    _stats,
    _transform,
    _variable,
)


class Backend:
    def __init__(self):
        self._fact = None
        self._fact_name = None

    def select(self, which, **kwargs):
        if self._fact is not None:
            raise ValueError(f"Factorisation {self._fact} has been selected already.")
        self._fact = _factorisations.choose(which, **kwargs)
        self._fact_name = which

    @property
    def impl_name(self):
        return self._fact_name

    @property
    def linearise(self) -> _linearise.LinearisationBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.linearise()

    @property
    def conditional(self) -> _conditional.ConditionalBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.conditional()

    @property
    def transform(self) -> _transform.TransformBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.transform()

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.ssm_util()

    @property
    def prototypes(self) -> _prototypes.PrototypeBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.prototypes()

    @property
    def variable(self) -> _variable.VariableBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.variable()

    @property
    def hidden_model(self) -> _hidden_model.HiddenModelBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.hidden_model()

    @property
    def stats(self) -> _stats.StatsBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.stats()
