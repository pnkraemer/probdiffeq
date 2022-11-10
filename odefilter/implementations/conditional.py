"""Conditionals."""
from typing import Any, Generic, TypeVar

import jax

SSVTypeVar = TypeVar("SSVTypeVar")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class BackwardModel(Generic[SSVTypeVar]):
    """Backward model for backward-Gauss--Markov process representations."""

    def __init__(self, *, transition: Any, noise: SSVTypeVar):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)

    def scale_covariance(self, *, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return BackwardModel(transition=self.transition, noise=noise)
