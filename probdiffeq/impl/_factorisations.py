import abc

from probdiffeq.impl import (
    _conditional,
    _linearise,
    _prototypes,
    _random,
    _ssm_util,
    _transform,
)


class Factorisation(abc.ABC):
    @abc.abstractmethod
    def linearise(self) -> _linearise.LinearisationBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def random(self) -> _random.RandomVariableBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self) -> _transform.TransformBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def conditional(self) -> _conditional.ConditionalBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def ssm_util(self) -> _ssm_util.SSMUtilBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def prototypes(self) -> _prototypes.PrototypeBackend:
        raise NotImplementedError


def choose(which, /, *, ode_shape=None):
    # In this function, we import outside toplevel.
    #
    # Why?
    # 1. To avoid cyclic imports
    # 2. To avoid import errors if some backends require additional dependencies
    # 3. To keep the import-namespace clean
    #    (factorisations.ScalarFactorisation is easier to read than
    #     probdiffeq.impl.scalar.factorisations.ScalarFactorisation())
    #
    if which == "scalar":
        from probdiffeq.impl.scalar import factorisations

        return factorisations.ScalarFactorisation()

    if ode_shape is None:
        raise ValueError("Please provide an ODE shape.")
    if which == "dense":
        from probdiffeq.impl.dense import factorisations

        return factorisations.DenseFactorisation(ode_shape=ode_shape)
    if which == "isotropic":
        from probdiffeq.impl.isotropic import factorisations

        return factorisations.IsotropicFactorisation(ode_shape=ode_shape)
    if which == "blockdiag":
        from probdiffeq.impl.blockdiag import factorisations

        return factorisations.BlockDiagFactorisation(ode_shape=ode_shape)

    raise ValueError
