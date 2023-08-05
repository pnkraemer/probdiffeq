"""State-space model backend."""

from probdiffeq.statespace.backend import cond, error, factorisations, linearise, rv


class BackEnd:
    def __init__(self):
        self._fact: factorisations.Factorisation = None

    def select(self, which):
        self._fact = factorisations.choose(which)

    @property
    def linearise_ode(self) -> linearise.LineariseODEBackEnd:
        return self._fact.linearise_ode

    @property
    def rv(self) -> rv.RandomVariableBackEnd:
        return self._fact.rv

    @property
    def cond(self) -> cond.ConditionalBackEnd:
        return self._fact.cond

    @property
    def error(self) -> error.ErrorBackEnd:
        return self._fact.error
