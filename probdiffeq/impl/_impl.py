"""State-space model impl."""
import warnings

from probdiffeq.backend import abc
from probdiffeq.backend.typing import Optional
from probdiffeq.impl import (
    _conditional,
    _hidden_model,
    _linearise,
    _prototypes,
    _ssm_util,
    _stats,
    _transform,
    _variable,
)


class FactorisedImpl(abc.ABC):
    """Interface for the implementations provided by the backend."""

    @abc.abstractmethod
    def linearise(self) -> _linearise.LinearisationBackend:
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

    @abc.abstractmethod
    def variable(self) -> _variable.VariableBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def hidden_model(self) -> _hidden_model.HiddenModelBackend:
        raise NotImplementedError

    @abc.abstractmethod
    def stats(self) -> _stats.StatsBackend:
        raise NotImplementedError


def choose(which: str, /, *, ode_shape=None) -> FactorisedImpl:
    # In this function, we import outside toplevel.
    #
    # Why?
    # 1. To avoid cyclic imports
    # 2. To avoid import errors if some backends require additional dependencies
    #
    if which == "scalar":
        import probdiffeq.impl.scalar.factorised_impl

        return probdiffeq.impl.scalar.factorised_impl.Scalar()

    if ode_shape is None:
        msg = "Please provide an ODE shape."
        raise ValueError(msg)

    if which == "dense":
        import probdiffeq.impl.dense.factorised_impl

        return probdiffeq.impl.dense.factorised_impl.Dense(ode_shape=ode_shape)

    if which == "isotropic":
        import probdiffeq.impl.isotropic.factorised_impl

        return probdiffeq.impl.isotropic.factorised_impl.Isotropic(ode_shape=ode_shape)

    if which == "blockdiag":
        import probdiffeq.impl.blockdiag.factorised_impl

        return probdiffeq.impl.blockdiag.factorised_impl.BlockDiag(ode_shape=ode_shape)
    msg1 = f"Implementation '{which}' unknown. "
    msg2 = "Choose an implementation out of {scalar, dense, isotropic, blockdiag}."
    raise ValueError(msg1 + msg2)


class Impl:
    """User-facing implementation 'package'.

    Wrap a factorised implementations and garnish it with error messages
    and a "selection" functionality.
    """

    def __init__(self) -> None:
        self._fact: Optional[FactorisedImpl] = None
        self._fact_name: str = "None"

    def select(self, which, **kwargs):
        if self._fact is not None:
            msg = f"An implementation has already been selected: '{self._fact_name}'."
            warnings.warn(msg, stacklevel=1)
        self._fact = choose(which, **kwargs)
        self._fact_name = which

    @property
    def impl_name(self):
        return self._fact_name

    @property
    def linearise(self) -> _linearise.LinearisationBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.linearise()

    @property
    def conditional(self) -> _conditional.ConditionalBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.conditional()

    @property
    def transform(self) -> _transform.TransformBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.transform()

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.ssm_util()

    @property
    def prototypes(self) -> _prototypes.PrototypeBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.prototypes()

    @property
    def variable(self) -> _variable.VariableBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.variable()

    @property
    def hidden_model(self) -> _hidden_model.HiddenModelBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.hidden_model()

    @property
    def stats(self) -> _stats.StatsBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.stats()
