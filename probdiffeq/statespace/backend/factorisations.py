import abc

import probdiffeq.statespace.backend.scalar.cond
import probdiffeq.statespace.backend.scalar.linearise
import probdiffeq.statespace.backend.scalar.random
from probdiffeq.statespace.backend import cond, linearise, random


class Factorisation(abc.ABC):
    @property
    @abc.abstractmethod
    def linearise_ode(self) -> linearise.LineariseODEBackEnd:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def random(self) -> random.RandomVariableBackEnd:
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
    def random(self):
        return probdiffeq.statespace.backend.scalar.random.RandomVariableBackEnd()

    @property
    def cond(self):
        return probdiffeq.statespace.backend.scalar.cond.ConditionalBackEnd()


def choose(which, /, **kwargs) -> Factorisation:
    if which == "scalar":
        return ScalarFactorisation(**kwargs)
    raise ValueError
