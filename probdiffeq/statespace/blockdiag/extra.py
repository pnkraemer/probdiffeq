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
    fi, sm, fp = ibm_stack._filter, ibm_stack._smoother, ibm_stack._fixedpoint
    dynamic, static = ibm_stack._dynamic, ibm_stack._static

    return _extra.ExtrapolationBundle(
        _blockdiag(fi),
        _blockdiag(sm),
        _blockdiag(fp),
        *dynamic,
        **static,
    )


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.concatenate([s[None, ...]] * n), tree)


def _blockdiag(ex: _extra.Extrapolation):
    def custom_constructor(*args, **kwargs):
        return _BlockDiag(ex(*args, **kwargs))

    return custom_constructor


# @jax.tree_util.register_pytree_node_class
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

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        solution_fn = jax.vmap(type(self.extra).solution_from_tcoeffs)
        return solution_fn(self.extra, taylor_coefficients)

    def init(self, sol, /):
        solution_fn = jax.vmap(type(self.extra).init)
        return solution_fn(self.extra, sol)

    def begin(self, ssv, extra, /, dt):
        fn = jax.vmap(type(self.extra).begin, in_axes=(0, 0, 0, None))
        return fn(self.extra, ssv, extra, dt)

    def complete(self, ssv, extra, /, output_scale):
        fn = jax.vmap(type(self.extra).complete)
        return fn(self.extra, ssv, extra, output_scale)

    def reset(self, ssv, extra, /):
        init_fn = jax.vmap(type(self.extra).reset)
        return init_fn(self.extra, ssv, extra)

    def extract(self, ssv, extra, /):
        # If called in save-at mode, batch again.
        if ssv.hidden_state.mean.ndim > 2:
            return jax.vmap(self.extract)(ssv, extra)

        solution_fn = jax.vmap(type(self.extra).extract)
        return solution_fn(self.extra, ssv, extra)

    # todo: move to correction?
    def promote_output_scale(self, output_scale):
        # broadcast to shape (d,)
        output_scale = output_scale * jnp.ones(self.ode_shape)
        fn_vmap = jax.vmap(type(self.extra).promote_output_scale)
        return fn_vmap(self.extra, output_scale)

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 1:
            return output_scale[-1, :]
        return output_scale
