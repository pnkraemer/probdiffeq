"""Various interfaces."""

from typing import Generic, Tuple, TypeVar

import jax

from probdiffeq.statespace import _collections

S = TypeVar("S", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("C")
"""A type-variable to alias correction-caches."""


@jax.tree_util.register_pytree_node_class
class Correction(Generic[S, C]):
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

    def begin(self, ssv: S, corr: C, /, vector_field, t, p) -> Tuple[S, C]:
        raise NotImplementedError

    def complete(self, ssv: S, corr: C, /, vector_field, t, p) -> Tuple[S, C]:
        raise NotImplementedError
