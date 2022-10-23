"""Information operator interfaces."""

import abc

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Information(abc.ABC):
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
    def linearise(self, x, /, *, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def cov(self, *, cache_obs, cov_sqrtm_lower):
        raise NotImplementedError
