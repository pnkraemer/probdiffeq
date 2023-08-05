import abc

import probdiffeq.statespace.backend.scalar.cond
import probdiffeq.statespace.backend.scalar.linearise
import probdiffeq.statespace.backend.scalar.random
import probdiffeq.statespace.backend.scalar.ssm_util
from probdiffeq.statespace.backend import _cond, _linearise, _random, _ssm_util


class Factorisation(abc.ABC):
    @abc.abstractmethod
    def linearise_ode(self) -> _linearise.LineariseODEBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def random(self) -> _random.RandomVariableBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def cond(self) -> _cond.ConditionalBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def ssm_util(self) -> _ssm_util.SSMUtilBackEnd:
        raise NotImplementedError


class ScalarFactorisation(Factorisation):
    def linearise_ode(self):
        return probdiffeq.statespace.backend.scalar.linearise.LineariseODEBackEnd()

    def random(self):
        return probdiffeq.statespace.backend.scalar.random.RandomVariableBackEnd()

    def cond(self):
        return probdiffeq.statespace.backend.scalar.cond.ConditionalBackEnd()

    def ssm_util(self):
        return probdiffeq.statespace.backend.scalar.ssm_util.SSMUtilBackEnd()


def choose(which, /, **kwargs) -> Factorisation:
    if which == "scalar":
        return ScalarFactorisation(**kwargs)
    raise ValueError
