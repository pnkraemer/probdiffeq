"""IVP-solver API."""


import abc
from typing import Generic, TypeVar

from probdiffeq import _interp

T = TypeVar("T")
"""A type-variable for state-types."""


class Solver(abc.ABC, Generic[T]):
    """IVP solver."""

    def __init__(self, strategy, *, string_repr, requires_rescaling):
        self.strategy = strategy

        self.string_repr = string_repr
        self.requires_rescaling = requires_rescaling

    def __repr__(self):
        return self.string_repr

    def solution_from_tcoeffs(self, tcoeffs, /, t, output_scale, num_steps=1.0):
        """Construct an initial `Solution` object."""
        posterior = self.strategy.solution_from_tcoeffs(tcoeffs)
        return t, posterior, output_scale, num_steps

    @abc.abstractmethod
    def init(self, t, posterior, /, output_scale, num_steps) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: T, *, vector_field, dt, parameters) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: T, /):
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, t, s0: T, s1: T) -> _interp.InterpRes[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def right_corner(self, t, s0: T, s1: T) -> _interp.InterpRes[T]:
        raise NotImplementedError
