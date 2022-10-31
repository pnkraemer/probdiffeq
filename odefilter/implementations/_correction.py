"""Interface for implementations."""

import abc
from typing import Generic, Tuple, TypeVar

import jax.numpy as jnp
import jax.tree_util
from jax import Array

R = TypeVar("R")  # think: Random variable type
C = TypeVar("C")  # think: Information operator's personal cache-type


@jax.tree_util.register_pytree_node_class
class Correction(abc.ABC, Generic[R, C]):
    """Correction model interface."""

    def __init__(self, *, ode_order=1):
        self.ode_order = ode_order

    def __repr__(self):
        return f"{self.__class__.__name__}(ode_order={self.ode_order})"

    def __eq__(self, other):
        equal = jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), self, other)
        return jax.tree_util.tree_all(equal)

    def tree_flatten(self):
        children = ()
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        (ode_order,) = aux
        return cls(ode_order=ode_order)

    @abc.abstractmethod
    def begin_correction(
        self, x: R, /, *, vector_field, t, p
    ) -> Tuple[Array, float, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, extrapolated: R, cache: C):
        raise NotImplementedError

    @abc.abstractmethod
    def evidence_sqrtm(self, *, observed: R):
        raise NotImplementedError
