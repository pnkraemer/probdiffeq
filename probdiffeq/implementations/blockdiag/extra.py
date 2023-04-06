"""Extrapolations."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections
from probdiffeq.implementations.scalar import extra as scalar_extra

_IBMCacheType = Tuple[jax.Array]  # Cache type
"""Type-variable for the extrapolation-cache."""


def ibm_blockdiag(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    (n,) = ode_shape
    ibm = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
    ibm_stack = _tree_stack_duplicates(ibm, n=n)
    return _BlockDiag(ibm_stack)


@jax.tree_util.register_pytree_node_class
class _BlockDiag(_collections.AbstractExtrapolation):
    def __init__(self, ibm):
        self.ibm = ibm

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(ibm={self.ibm})"

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
        return cls(ibm)

    def begin_extrapolation(self, p0, /, dt):
        fn = jax.vmap(scalar_extra._IBM.begin_extrapolation, in_axes=(0, 0, None))
        return fn(self.ibm, p0, dt)

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        fn = jax.vmap(scalar_extra._IBM.complete_extrapolation)
        return fn(self.ibm, linearisation_pt, p0, cache, output_scale_sqrtm)

    def init_conditional(self, ssv_proto):
        return jax.vmap(scalar_extra._IBM.init_conditional)(self.ibm, ssv_proto)

    def init_hidden_state(self, taylor_coefficients):
        return jax.vmap(scalar_extra._IBM.init_hidden_state)(
            self.ibm, taylor_coefficients
        )

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(scalar_extra._IBM.init_error_estimate)(self.ibm)

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jax.vmap(scalar_extra._IBM.init_output_scale_sqrtm)(self.ibm)

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        fn = jax.vmap(scalar_extra._IBM.revert_markov_kernel)
        return fn(self.ibm, linearisation_pt, p0, cache, output_scale_sqrtm)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.concatenate([s[None, ...]] * n), tree)
