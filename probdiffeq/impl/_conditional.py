"""Conditionals."""
from probdiffeq.backend import abc


class ConditionalBackend(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, cond1, cond2, /):
        raise NotImplementedError
