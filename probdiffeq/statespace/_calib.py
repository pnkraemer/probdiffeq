"""Calibration API."""


import abc
from typing import Callable

from probdiffeq.backend import containers


class Calibration(containers.NamedTuple):
    init: Callable
    update: Callable
    extract: Callable


# See _extra.ExtrapolationFactory for what is going on here.
# TL;DR: calibration implementations are tied to state-space model factorisations,
# but at the time of choosing the factorisation, it is too early to choose a method.
# The factory below allows delaying this decision to later on.
class CalibrationFactory(abc.ABC):
    @abc.abstractmethod
    def mle(self) -> Calibration:
        raise NotImplementedError

    @abc.abstractmethod
    def dynamic(self) -> Calibration:
        raise NotImplementedError

    @abc.abstractmethod
    def free(self) -> Calibration:
        raise NotImplementedError
