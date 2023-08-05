import abc

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


def choose(which, /):
    if which == "scalar":
        # Import outside toplevel.
        # Why?
        # 1. To avoid cyclic imports
        # 2. To avoid import errors if some backends require additional dependencies
        # 3. To keep the import-namespace clean
        #    (factorisations.ScalarFactorisation is easier to read than
        #     probdiffeq.statespace.backend.scalar.factorisations.ScalarFactorisation())
        from probdiffeq.statespace.backend.scalar import factorisations

        return factorisations.ScalarFactorisation()

    raise ValueError
