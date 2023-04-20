"""Various interfaces."""

from typing import Generic, Tuple, TypeVar

import jax

from probdiffeq.statespace import _collections

S = TypeVar("S", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("C")
"""A type-variable to alias extrapolation-caches."""


@jax.tree_util.register_pytree_node_class
class ExtrapolationBundle:
    def __init__(self, filter, smoother, fixedpoint, **kwargs):
        self._filter = filter
        self._smoother = smoother
        self._fixedpoint = fixedpoint
        self.kwargs = kwargs

    def tree_flatten(self):
        children = (self.kwargs,)
        aux = self._filter, self._smoother, self._fixedpoint
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        filter, smoother, fixedpoint = aux
        (kwargs,) = children
        return cls(filter, smoother, fixedpoint, **kwargs)

    @property
    def num_derivatives(self):
        return self.filter.num_derivatives

    @property
    def filter(self):
        return self._filter(**self.kwargs)

    @property
    def smoother(self):
        return self._smoother(**self.kwargs)

    @property
    def fixedpoint(self):
        return self._fixedpoint(**self.kwargs)


class Extrapolation(Generic[S, C]):
    """Extrapolation model interface."""

    def __init__(self, a, q_sqrtm_lower, preconditioner_scales, preconditioner_powers):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

        self.preconditioner_scales = preconditioner_scales
        self.preconditioner_powers = preconditioner_powers

    def tree_flatten(self):
        children = (
            self.a,
            self.q_sqrtm_lower,
            self.preconditioner_scales,
            self.preconditioner_powers,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower, scales, powers = children
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm_lower,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def promote_output_scale(self, output_scale) -> float:
        raise NotImplementedError

    def solution_from_tcoeffs(self, taylor_coefficients, /) -> S:
        raise NotImplementedError

    def filter_begin(self, ssv: S, extra: C, /, dt) -> Tuple[S, C]:
        raise NotImplementedError

    def smoother_begin(self, ssv: S, extra: C, /, dt) -> Tuple[S, C]:
        raise NotImplementedError

    def filter_complete(self, ssv: S, extra: C, /, output_scale) -> Tuple[S, C]:
        raise NotImplementedError

    def smoother_complete(self, ssv: S, extra: C, /, output_scale) -> Tuple[S, C]:
        raise NotImplementedError

    # todo: bundle in an init() method:

    def init_error_estimate(self) -> jax.Array:
        raise NotImplementedError

    def smoother_init_conditional(self, ssv_proto):
        raise NotImplementedError
