"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _collections
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
class _BlockDiag(_collections.AbstractExtrapolation):
    def __init__(self, extra, /):
        # todo: init of superclass?
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

    def solution_from_tcoeffs_without_reversal(self, taylor_coefficients, /):
        solution_fn = jax.vmap(type(self.extra).solution_from_tcoeffs_without_reversal)
        return solution_fn(self.extra, taylor_coefficients)

    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /):
        solution_fn = jax.vmap(type(self.extra).solution_from_tcoeffs_with_reversal)
        return solution_fn(self.extra, taylor_coefficients)

    def init_without_reversal(self, rv, /):
        solution_fn = jax.vmap(type(self.extra).init_without_reversal)
        return solution_fn(self.extra, rv)

    def init_with_reversal(self, rv, conds, /):
        solution_fn = jax.vmap(type(self.extra).init_with_reversal)
        return solution_fn(self.extra, rv, conds)

    def extract_with_reversal(self, s, /):
        solution_fn = jax.vmap(type(self.extra).extract_with_reversal)

        # In save-at-mode (i.e. not terminal values), the vmapping requires extra
        #  vmapping in the `s` variable.
        if s.hidden_state.mean.ndim > 2:
            solution_fn = jax.vmap(solution_fn, in_axes=(None, 0))

        return solution_fn(self.extra, s)

    def extract_without_reversal(self, s, /):
        solution_fn = jax.vmap(type(self.extra).extract_without_reversal)

        # In save-at-mode (i.e. not terminal values), the vmapping requires extra
        #  vmapping in the `s` variable.
        if s.hidden_state.mean.ndim > 2:
            solution_fn = jax.vmap(solution_fn, in_axes=(None, 0))

        return solution_fn(self.extra, s)

    def begin(self, s0, /, dt):
        fn = jax.vmap(type(self.extra).begin, in_axes=(0, 0, None))
        return fn(self.extra, s0, dt)

    def complete_without_reversal(self, state, /, state_previous, output_scale):
        fn = type(self.extra).complete_without_reversal
        fn_vmap = jax.vmap(fn)
        return fn_vmap(self.extra, state, state_previous, output_scale)

    def init_conditional(self, ssv_proto):
        return jax.vmap(type(self.extra).init_conditional)(self.extra, ssv_proto)

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(type(self.extra).init_error_estimate)(self.extra)

    # todo: move to correction?
    def promote_output_scale(self, output_scale):
        # broadcast to shape (d,)
        output_scale = output_scale * jnp.ones(self.ode_shape)
        fn_vmap = jax.vmap(type(self.extra).promote_output_scale)
        return fn_vmap(self.extra, output_scale)

    def complete_with_reversal(self, state, /, state_previous, output_scale):
        fn = jax.vmap(type(self.extra).complete_with_reversal)
        return fn(self.extra, state, state_previous, output_scale)

    def replace_backward_model(self, s, /, backward_model):
        fn = jax.vmap(type(self.extra).replace_backward_model)
        return fn(self.extra, s, backward_model)
