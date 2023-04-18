"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _extra
from probdiffeq.statespace.scalar import extra as scalar_extra


def ibm_blockdiag(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    (n,) = ode_shape
    ibm = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
    ibm_stack = _tree_stack_duplicates(ibm, n=n)
    return _BlockDiag(ibm_stack)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.concatenate([s[None, ...]] * n), tree)


@jax.tree_util.register_pytree_node_class
class _BlockDiag(_extra.Extrapolation):
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

    def filter_begin(self, ssv, extra, /, dt):
        fn = jax.vmap(type(self.extra).filter_begin, in_axes=(0, 0, 0, None))
        return fn(self.extra, ssv, extra, dt)

    def filter_complete(self, ssv, extra, /, output_scale):
        fn = jax.vmap(type(self.extra).filter_complete)
        return fn(self.extra, ssv, extra, output_scale)

    def init_conditional(self, ssv_proto):
        return jax.vmap(type(self.extra).init_conditional)(self.extra, ssv_proto)

    def filter_solution_from_tcoeffs(self, taylor_coefficients, /):
        solution_fn = jax.vmap(type(self.extra).filter_solution_from_tcoeffs)
        return solution_fn(self.extra, taylor_coefficients)

    def filter_init(self, sol, /):
        solution_fn = jax.vmap(type(self.extra).filter_init)
        return solution_fn(self.extra, sol)

    def filter_extract(self, ssv, extra, /):
        # If called in save-at mode, batch again.
        if ssv.hidden_state.mean.ndim > 2:
            return jax.vmap(self.filter_extract)(ssv, extra)

        solution_fn = jax.vmap(type(self.extra).filter_extract)
        return solution_fn(self.extra, ssv, extra)

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(type(self.extra).init_error_estimate)(self.extra)

    # todo: move to correction?
    def promote_output_scale(self, output_scale):
        # broadcast to shape (d,)
        output_scale = output_scale * jnp.ones(self.ode_shape)
        fn_vmap = jax.vmap(type(self.extra).promote_output_scale)
        return fn_vmap(self.extra, output_scale)

    def complete_with_reversal(self, output_begin, /, s0, output_scale):
        fn = jax.vmap(type(self.extra).complete_with_reversal)
        return fn(self.extra, output_begin, s0, output_scale)
