"""PyTree utilities.

Extremely experimental. Might delete tomorrow.
"""


import jax
import jax.numpy as jnp


class TreeEqualMixIn:
    def __eq__(self, other):
        equal = jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), self, other)
        return jax.tree_util.tree_all(equal)


class TreeShapeMixIn:
    def tree_shape(self):
        return jax.tree_util.tree_map(jnp.shape, self)


class TreeNDimMixIn:
    def tree_ndim(self):
        return jax.tree_util.tree_map(jnp.ndim, self)
