"""Various interfaces."""

import abc
from typing import Any, Generic, NamedTuple, Tuple, TypeVar

import jax

from probdiffeq.statespace import _collections


class State(NamedTuple):
    observed: Any
    error_estimate: Any
    output_scale_dynamic: Any
    cache: Any

    def scale_covariance(self, s, /):
        observed = self.observed.scale_covariance(s)
        return State(
            observed,
            error_estimate=self.error_estimate,
            output_scale_dynamic=self.output_scale_dynamic,
            cache=self.cache,
        )


S = TypeVar("S", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("S", bound=State)


@jax.tree_util.register_pytree_node_class
class Correction(abc.ABC, Generic[S]):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    def tree_flatten(self):
        children = ()
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        (ode_order,) = aux
        return cls(ode_order=ode_order)

    @abc.abstractmethod
    def init(self, s: S, /) -> Tuple[S, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, s: S, c: C, /, vector_field, t, p) -> Tuple[S, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, s: S, c: C, /, vector_field, t, p) -> Tuple[S, C]:
        raise NotImplementedError
