"""Correction-model API."""

import abc
from typing import Generic, Tuple, TypeVar

from probdiffeq.statespace import variables

S = TypeVar("S", bound=variables.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("C")
"""A type-variable to alias correction-caches."""


class Correction(Generic[S, C], abc.ABC):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    @abc.abstractmethod
    def init(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_error(self, ssv: S, corr: C, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv: S, corr: C, /, vector_field, t, p) -> Tuple[S, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, corr, /):
        raise NotImplementedError

    #
    #
    #
    #
    #
    #
    #
    #
    # todo: delete

    def tree_flatten(self):
        children = ()
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        (ode_order,) = aux
        return cls(ode_order=ode_order)
