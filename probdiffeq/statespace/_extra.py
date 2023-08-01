"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

from probdiffeq.statespace import variables

S = TypeVar("S", bound=variables.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("C")
"""A type-variable to alias extrapolation-caches."""


class ExtrapolationFactory(abc.ABC):
    @abc.abstractmethod
    def filter(self, *params):
        raise NotImplementedError

    @abc.abstractmethod
    def smoother(self, *params):
        raise NotImplementedError

    @abc.abstractmethod
    def fixedpoint(self, *params):
        raise NotImplementedError


# @jax.tree_util.register_pytree_node_class
# class ExtrapolationBundle:
#     def __init__(self, filter, smoother, fixedpoint, *dynamic, **static):
#         self._filter = filter
#         self._smoother = smoother
#         self._fixedpoint = fixedpoint
#         self._dynamic = dynamic
#         self._static = static
#
#     def __repr__(self):
#         return repr(self.filter)
#
#     def tree_flatten(self):
#         children = (self._dynamic,)
#         aux = self._filter, self._smoother, self._fixedpoint, self._static
#         return children, aux
#
#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         filter, smoother, fixedpoint, static = aux
#         (dynamic,) = children
#         return cls(filter, smoother, fixedpoint, *dynamic, **static)
#
#     @property
#     def num_derivatives(self):
#         return self.filter.num_derivatives
#
#     @property
#     def filter(self):
#         return self._filter(*self._dynamic, **self._static)
#
#     @property
#     def smoother(self):
#         return self._smoother(*self._dynamic, **self._static)
#
#     @property
#     def fixedpoint(self):
#         return self._fixedpoint(*self._dynamic, **self._static)


class Extrapolation(Generic[S, C]):
    """Extrapolation model interface."""

    def __init__(self, a, q_sqrtm_lower, preconditioner):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower
        self.preconditioner = preconditioner

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def solution_from_tcoeffs(self, taylor_coefficients, /) -> S:
        raise NotImplementedError

    def begin(self, ssv: S, extra: C, /, dt) -> Tuple[S, C]:
        raise NotImplementedError

    def complete(self, ssv: S, extra: C, /, output_scale) -> Tuple[S, C]:
        raise NotImplementedError

    def init(self, sol, /):
        raise NotImplementedError

    def extract(self, ssv, extra, /):
        raise NotImplementedError
