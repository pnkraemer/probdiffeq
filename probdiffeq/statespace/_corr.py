"""Correction-model API."""

import abc


class Correction(abc.ABC):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    @abc.abstractmethod
    def init(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, corr, /):
        raise NotImplementedError
