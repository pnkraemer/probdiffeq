"""Batch-style implementations."""
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from odefilter.implementations import _ibm, _implementation

# todo: reconsider naming!


@dataclass(frozen=True)
class BatchImplementation(_implementation.Implementation):
    """Handle block-diagonal covariances."""

    a: Any
    q_sqrtm_lower: Any

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        a, q_sqrtm_lower = children
        n, d = aux
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower, num_derivatives=n, ode_dimension=d)

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm.system_matrices_1d(num_derivatives=num_derivatives)
        a = jnp.stack([a] * ode_dimension)
        q_sqrtm = jnp.stack([q_sqrtm] * ode_dimension)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)
