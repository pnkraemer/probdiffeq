"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections
from probdiffeq.implementations.scalar import extra as scalar_extra


def ibm_blockdiag(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    (n,) = ode_shape
    ibm = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
    ibm_stack = _tree_stack_duplicates(ibm, n=n)
    return _BlockDiag(ibm_stack)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.concatenate([s[None, ...]] * n), tree)


@jax.tree_util.register_pytree_node_class
class _BlockDiag(_collections.AbstractExtrapolation):
    def __init__(self, extra, /):
        self.extra = extra

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}({self.extra})"

    @property
    def num_derivatives(self):
        return self.extra.a.shape[1] - 1

    @property
    def ode_shape(self):
        # todo: this does not scale to matrix-valued problems
        return self.extra.a.shape[0]

    def tree_flatten(self):
        children = (self.extra,)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (extra,) = children
        return cls(extra)

    def begin_extrapolation(self, p0, /, dt):
        fn = jax.vmap(type(self.extra).begin_extrapolation, in_axes=(0, 0, None))
        return fn(self.extra, p0, dt)

    def complete_extrapolation_without_reversal(
        self, linearisation_pt, p0, output_scale
    ):
        fn = jax.vmap(type(self.extra).complete_extrapolation_without_reversal)
        return fn(self.extra, linearisation_pt, p0, output_scale)

    def init_conditional(self, ssv_proto):
        return jax.vmap(type(self.extra).init_conditional)(self.extra, ssv_proto)

    def init_state_space_var(self, taylor_coefficients):
        return jax.vmap(type(self.extra).init_state_space_var)(
            self.extra, taylor_coefficients
        )

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(type(self.extra).init_error_estimate)(self.extra)

    # todo: move to correction?
    def init_output_scale(self, output_scale):
        # broadcast to shape (d,)
        output_scale = output_scale * jnp.ones(self.ode_shape)
        fn_vmap = jax.vmap(type(self.extra).init_output_scale)
        return fn_vmap(self.extra, output_scale)

    def complete_extrapolation_with_reversal(self, linearisation_pt, p0, output_scale):
        fn = jax.vmap(type(self.extra).complete_extrapolation_with_reversal)
        return fn(self.extra, linearisation_pt, p0, output_scale)
