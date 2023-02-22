"""Extrapolations."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _ibm_util, _scalar

_IBMCacheType = Tuple[jax.Array]  # Cache type
"""Type-variable for the extrapolation-cache."""


@jax.tree_util.register_pytree_node_class
class BlockDiagIBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower):
        self.ibm = _scalar.IBM(a, q_sqrtm_lower)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.ibm.a}, q_sqrtm_lower={self.ibm.q_sqrtm_lower})"

    @property
    def num_derivatives(self):
        return self.ibm.a.shape[1] - 1

    @property
    def ode_shape(self):
        # todo: this does not scale to matrix-valued problems
        return self.ibm.a.shape[0]

    def tree_flatten(self):
        children = (self.ibm,)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (ibm,) = children
        return cls(a=ibm.a, q_sqrtm_lower=ibm.q_sqrtm_lower)

    @classmethod
    def from_params(cls, ode_shape, num_derivatives):
        assert len(ode_shape) == 1
        (n,) = ode_shape
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a_stack, q_sqrtm_stack = _tree_stack_duplicates((a, q_sqrtm), n=n)
        return cls(a=a_stack, q_sqrtm_lower=q_sqrtm_stack)

    def begin_extrapolation(self, p0, /, dt):
        fn = jax.vmap(_scalar.IBM.begin_extrapolation, in_axes=(0, 0, None))
        return fn(self.ibm, p0, dt)

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        fn = jax.vmap(_scalar.IBM.complete_extrapolation)
        return fn(self.ibm, linearisation_pt, p0, cache, output_scale_sqrtm)

    def init_conditional(self, ssv_proto):
        return jax.vmap(_scalar.IBM.init_conditional)(self.ibm, ssv_proto)

    def init_corrected(self, taylor_coefficients):
        return jax.vmap(_scalar.IBM.init_corrected)(self.ibm, taylor_coefficients)

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(_scalar.IBM.init_error_estimate)(self.ibm)

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jax.vmap(_scalar.IBM.init_output_scale_sqrtm)(self.ibm)

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        fn = jax.vmap(_scalar.IBM.revert_markov_kernel)
        return fn(self.ibm, linearisation_pt, p0, cache, output_scale_sqrtm)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.vstack([s[None, ...]] * n), tree)
