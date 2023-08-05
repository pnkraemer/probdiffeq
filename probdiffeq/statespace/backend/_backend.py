"""State-space model backend."""

from probdiffeq.statespace.backend import (
    _cond,
    _linearise,
    _random,
    _ssm_util,
    factorisations,
)


class BackEnd:
    def __init__(self):
        self._fact: factorisation.Factorisation = None

    def select(self, which, **kwargs):
        self._fact = factorisations.choose(which, **kwargs)

    @property
    def linearise_ode(self) -> _linearise.LineariseODEBackEnd:
        return self._fact.linearise_ode()

    @property
    def random(self) -> _random.RandomVariableBackEnd:
        return self._fact.random()

    @property
    def cond(self) -> _cond.ConditionalBackEnd:
        return self._fact.cond()

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackEnd:
        return self._fact.ssm_util()
