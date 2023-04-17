"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

import jax

from probdiffeq.statespace import _collections

S = TypeVar("S", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""

CacheTypeVar = TypeVar("CacheTypeVar")
"""A type-variable to alias extrapolation- and correction-caches."""


@jax.tree_util.register_pytree_node_class
class Correction(abc.ABC, Generic[S, CacheTypeVar]):
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
    def begin(
        self, x: S, /, vector_field, t, p
    ) -> Tuple[jax.Array, float, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, extrapolated: S, cache: CacheTypeVar):
        raise NotImplementedError
