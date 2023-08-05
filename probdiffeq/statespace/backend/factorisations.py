import abc

import probdiffeq.statespace.backend.scalar.cond
import probdiffeq.statespace.backend.scalar.error
import probdiffeq.statespace.backend.scalar.linearise
import probdiffeq.statespace.backend.scalar.rv
from probdiffeq.statespace.backend import cond, linearise, rv


class Factorisation(abc.ABC):
    @property
    @abc.abstractmethod
    def linearise_ode(self) -> linearise.LineariseODEBackEnd:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rv(self) -> rv.RandomVariableBackEnd:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cond(self) -> cond.ConditionalBackEnd:
        raise NotImplementedError


class ScalarFactorisation(Factorisation):
    @property
    def linearise_ode(self):
        return probdiffeq.statespace.backend.scalar.linearise.LineariseODEBackEnd()

    @property
    def rv(self):
        return probdiffeq.statespace.backend.scalar.rv.RandomVariableBackEnd()

    @property
    def cond(self):
        return probdiffeq.statespace.backend.scalar.cond.ConditionalBackEnd()

    @property
    def error(self):
        return probdiffeq.statespace.backend.scalar.error.ErrorBackEnd()


def choose(which, /) -> Factorisation:
    if which == "scalar":
        return ScalarFactorisation()
    raise ValueError
