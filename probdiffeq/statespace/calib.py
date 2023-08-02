"""Calibration API."""


import abc


class Calibration(abc.ABC):
    """Calibration implementation."""

    @abc.abstractmethod
    def init(self, prior):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, /, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state, /):
        raise NotImplementedError


class CalibrationFactory(abc.ABC):
    """Calibration factory.

    Calibration implementations are tied to state-space model factorisations,
    but at the time of choosing the factorisation, it is too early to choose a method.
    This factory allows delaying this decision to later.
    """

    @abc.abstractmethod
    def running_mean(self) -> Calibration:
        raise NotImplementedError

    @abc.abstractmethod
    def most_recent(self) -> Calibration:
        raise NotImplementedError
