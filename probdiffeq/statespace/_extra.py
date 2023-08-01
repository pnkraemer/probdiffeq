"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

from probdiffeq.statespace import variables

S = TypeVar("S", bound=variables.SSV)
"""A type-variable to alias appropriate state-space variable types."""

C = TypeVar("C")
"""A type-variable to alias extrapolation-caches."""


# At the point of choosing the recipe
# (aka selecting the desired state-space model factorisation),
# it is too early to know whether we solve forward-in-time only (aka filtering)
# or require a dense, or fixed-point solution. Therefore, the recipes return
# extrapolation *factories* instead of extrapolation models.
class ExtrapolationFactory(abc.ABC):
    @abc.abstractmethod
    def string_repr(self, *params):
        raise NotImplementedError

    @abc.abstractmethod
    def filter(self, *params):
        raise NotImplementedError

    @abc.abstractmethod
    def smoother(self, *params):
        raise NotImplementedError

    @abc.abstractmethod
    def fixedpoint(self, *params):
        raise NotImplementedError


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

    def init(self, sol, /):
        raise NotImplementedError

    def begin(self, ssv: S, extra: C, /, dt) -> Tuple[S, C]:
        raise NotImplementedError

    def complete(self, ssv: S, extra: C, /, output_scale) -> Tuple[S, C]:
        raise NotImplementedError

    def extract(self, ssv, extra, /):
        raise NotImplementedError
