"""IVP-solver API."""


from probdiffeq import _interp
from probdiffeq.backend import abc
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Generic, TypeVar
from probdiffeq.impl import impl

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

    def initial_condition(self, tcoeffs, /, output_scale):
        """Construct an initial condition."""
        if np.shape(output_scale) != np.shape(impl.prototypes.output_scale()):
            msg1 = "Argument 'output_scale' has the wrong shape. "
            msg2 = f"Shape {np.shape(impl.prototypes.output_scale())} expected; "
            msg3 = f"shape {np.shape(output_scale)} received."
            raise ValueError(msg1 + msg2 + msg3)
        posterior = self.strategy.initial_condition(tcoeffs)
        return posterior, output_scale

    @abc.abstractmethod
    def init(self, t, initial_condition) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: T, *, vector_field, dt) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: T, /):
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, t, s0: T, s1: T) -> _interp.InterpRes[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def right_corner(self, s0: T, s1: T) -> _interp.InterpRes[T]:
        raise NotImplementedError
