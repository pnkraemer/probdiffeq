import abc
from typing import Generic, TypeVar

import jax

# todo: solve this transform/derivative matrix/matmul dilemma

T = TypeVar("T")


class ConditionalBackend(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def marginalise(self, rv: T, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv: T, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x: T, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, cond1, cond2, /):
        raise NotImplementedError
