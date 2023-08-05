"""State-space model backend."""

from probdiffeq.statespace.backend import cond, factorisations, linearise, random


class BackEnd:
    def __init__(self):
        self._fact: factorisations.Factorisation = None

    def select(self, which, **kwargs):
        self._fact = factorisations.choose(which, **kwargs)

    @property
    def linearise_ode(self) -> linearise.LineariseODEBackEnd:
        return self._fact.linearise_ode

    @property
    def random(self) -> random.RandomVariableBackEnd:
        return self._fact.random

    @property
    def cond(self) -> cond.ConditionalBackEnd:
        return self._fact.cond
