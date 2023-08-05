"""State-space model backend."""

from probdiffeq.statespace.backend import factorisations, linearise


class BackEnd:
    def __init__(self):
        self._fact: factorisations.Factorisation = None

    def select(self, which):
        self._fact = factorisations.choose(which)

    @property
    def linearise_ode(self) -> linearise.LineariseODE:
        return self._fact.linearise_ode
