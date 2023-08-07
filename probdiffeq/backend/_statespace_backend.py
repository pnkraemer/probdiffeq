"""State-space model backend."""

from probdiffeq.backend import (
    _conditional,
    _factorisations,
    _linearise,
    _random,
    _ssm_util,
    _transform,
)


class BackEnd:
    def __init__(self):
        self._fact: _factorisations.Factorisation = None

    def select(self, which, **kwargs):
        self._fact = _factorisations.choose(which, **kwargs)

    @property
    def linearise_ode(self) -> _linearise.LineariseODEBackEnd:
        return self._fact.linearise_ode()

    @property
    def random(self) -> _random.RandomVariableBackEnd:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.random()

    @property
    def conditional(self) -> _conditional.ConditionalBackEnd:
        return self._fact.conditional()

    @property
    def transform(self) -> _transform.TransformBackEnd:
        return self._fact.transform()

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackEnd:
        return self._fact.ssm_util()
