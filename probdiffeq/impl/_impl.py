"""State-space model impl."""

from probdiffeq.impl import (
    _conditional,
    _factorisations,
    _linearise,
    _random,
    _ssm_util,
    _transform,
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
    def random(self) -> _random.RandomVariableBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.random()

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
    def prototypes(self) -> _ssm_util.SSMUtilBackend:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.prototypes()
