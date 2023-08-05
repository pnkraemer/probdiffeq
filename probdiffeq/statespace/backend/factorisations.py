import abc

import probdiffeq.statespace.backend.scalar.linearise
from probdiffeq.statespace.backend import linearise


class Factorisation(abc.ABC):
    @property
    @abc.abstractmethod
    def linearise_ode(self) -> linearise.LineariseODE:
        raise NotImplementedError


class ScalarFactorisation(Factorisation):
    @property
    def linearise_ode(self):
        return probdiffeq.statespace.backend.scalar.linearise.LineariseODE()


def choose(which, /) -> Factorisation:
    if which == "scalar":
        return ScalarFactorisation()
    raise ValueError
