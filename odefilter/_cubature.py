"""Cubature rules."""
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass
class _PositiveCubatureRule:
    """Cubature rule with positive weights."""

    points: Array
    weights_sqrtm: Array

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.points.shape[0]})"

    def tree_flatten(self):
        children = self.points, self.weights_sqrtm
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        pts, weights_sqrtm = children
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@register_pytree_node_class
@dataclass
class SCI(_PositiveCubatureRule):
    """Spherical cubature integration."""

    @classmethod
    def from_dimension(cls, *, dim):
        """Construct an SCI rule from the dimension of a random variable.

        The number of cubature points is _higher_ than ``dim``.
        """
        eye_d = jnp.eye(dim) * jnp.sqrt(dim)
        pts = jnp.vstack((eye_d, -1 * eye_d))
        weights_sqrtm = jnp.ones((2 * dim,)) / jnp.sqrt(2.0 * dim)
        return cls(points=pts, weights_sqrtm=weights_sqrtm)
