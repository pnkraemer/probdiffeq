"""Extrapolation model interfaces."""

import abc


# At the point of choosing the recipe
# (aka selecting the desired state-space model factorisation),
# it is too early to know whether we solve forward-in-time only (aka filtering)
# or require a dense, or fixed-point solution. Therefore, the recipes return
# extrapolation *factories* instead of extrapolation models.
class ExtrapolationFactory(abc.ABC):
    @abc.abstractmethod
    def string_repr(self):
        raise NotImplementedError

    @abc.abstractmethod
    def filter(self):
        raise NotImplementedError

    @abc.abstractmethod
    def smoother(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fixedpoint(self):
        raise NotImplementedError


class Extrapolation(abc.ABC):
    """Extrapolation model interface."""

    @abc.abstractmethod
    def solution_from_tcoeffs(self, taylor_coefficients, /):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, sol, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, ssv, extra, /, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, extra, /, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, extra, /):
        raise NotImplementedError
