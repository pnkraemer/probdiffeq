"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace import extra
from probdiffeq.statespace.scalar import extra as scalar_extra


def ibm_factory(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    fun_vmap = jax.vmap(scalar_extra.ibm_factory, in_axes=(None, 0))
    factory = fun_vmap(num_derivatives, jnp.ones(ode_shape))
    return _BlockDiagExtrapolationFactory(wraps=factory)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.concatenate([s[None, ...]] * n), tree)


class _BlockDiagExtrapolationFactory(extra.ExtrapolationFactory):
    def __init__(self, wraps):
        self.wraps = wraps

    def string_repr(self):
        num_derivatives = self.filter().num_derivatives
        ode_shape = self.filter().ode_shape
        args = f"num_derivatives={num_derivatives}, ode_shape={ode_shape}"
        return f"<Block-diagonal IBM with {args}>"

    def filter(self):
        return _BlockDiag(self.wraps.filter())

    def smoother(self):
        return _BlockDiag(self.wraps.smoother())

    def fixedpoint(self):
        return _BlockDiag(self.wraps.fixedpoint())


class _BlockDiag(extra.Extrapolation):
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
        if jnp.ndim(output_scale) == 0:
            raise ValueError(
                f"Output-scale with ndim={1} expected, "
                f"but output-scale {output_scale} with "
                f"ndim={jnp.ndim(output_scale)} received."
            )

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
