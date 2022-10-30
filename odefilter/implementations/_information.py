"""Interface for implementations."""

import abc

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Information(abc.ABC):
    """Interface for information operators."""

    def __init__(self, f, /, *, ode_order):
        self.f = f
        self.ode_order = ode_order

    def tree_flatten(self):
        children = ()
        aux = self.f, self.ode_order
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        f, ode_order = aux
        return cls(f, ode_order=ode_order)

    @abc.abstractmethod
    def begin_correction(self, x, /, *, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, extrapolated, cache):
        raise NotImplementedError

    @abc.abstractmethod
    def evidence_sqrtm(self, *, observed):
        raise NotImplementedError
