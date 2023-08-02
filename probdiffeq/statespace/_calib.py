"""Calibration API."""


import abc


class Calibration(abc.ABC):
    @abc.abstractmethod
    def init(self, prior):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, /, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state, /):
        raise NotImplementedError


# See _extra.ExtrapolationFactory for what is going on here.
# TL;DR: calibration implementations are tied to state-space model factorisations,
# but at the time of choosing the factorisation, it is too early to choose a method.
# The factory below allows delaying this decision to later on.
class CalibrationFactory(abc.ABC):
    @abc.abstractmethod
    def running_mean(self) -> Calibration:
        raise NotImplementedError

    @abc.abstractmethod
    def most_recent(self) -> Calibration:
        raise NotImplementedError
