"""IVP-solver API."""


import abc
from typing import Generic, TypeVar

import jax.numpy as jnp

from probdiffeq import _interp
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
        if jnp.shape(output_scale) != jnp.shape(impl.prototypes.output_scale()):
            msg1 = "Argument 'output_scale' has the wrong shape. "
            msg2 = f"Shape {jnp.shape(impl.prototypes.output_scale())} expected; "
            msg3 = f"shape {jnp.shape(output_scale)} received."
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
