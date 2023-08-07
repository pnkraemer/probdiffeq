import abc

from probdiffeq.backend import _conditional, _linearise, _random, _ssm_util, _transform


class Factorisation(abc.ABC):
    @abc.abstractmethod
    def linearise_ode(self) -> _linearise.LineariseODEBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def random(self) -> _random.RandomVariableBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self) -> _transform.TransformBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def conditional(self) -> _conditional.ConditionalBackEnd:
        raise NotImplementedError

    @abc.abstractmethod
    def ssm_util(self) -> _ssm_util.SSMUtilBackEnd:
        raise NotImplementedError


def choose(which, /, **options_for_dense_factorisation):
    if which == "scalar":
        # Import outside toplevel.
        # Why?
        # 1. To avoid cyclic imports
        # 2. To avoid import errors if some backends require additional dependencies
        # 3. To keep the import-namespace clean
        #    (_factorisations.ScalarFactorisation is easier to read than
        #     probdiffeq.backend.scalar._factorisations.ScalarFactorisation())
        from probdiffeq.backend.scalar import factorisations

        return factorisations.ScalarFactorisation()
    if which == "dense":
        from probdiffeq.backend.dense import factorisations

        return factorisations.DenseFactorisation(**options_for_dense_factorisation)
    if which == "isotropic":
        from probdiffeq.backend.isotropic import factorisations

        return factorisations.IsotropicFactorisation()

    raise ValueError
