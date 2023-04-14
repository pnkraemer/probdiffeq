"""Interface for estimation strategies."""

import abc
from typing import Generic, TypeVar

import jax

from probdiffeq._collections import InterpRes

S = TypeVar("S")
"""A type-variable to indicate strategy-state types."""

P = TypeVar("P")
"""A type-variable to indicate strategy-solution ("posterior") types."""

R = TypeVar("R")
"""A type-variable to indicate random-variable types."""


class Posterior(abc.ABC, Generic[R]):
    @abc.abstractmethod
    def sample(self, key, *, shape):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC, Generic[S, P]):
    """Inference strategy interface."""

    def __init__(self, extrapolation, correction):
        self.extrapolation = extrapolation
        self.correction = correction

    def __repr__(self):
        name = self.__class__.__name__
        arg1 = self.extrapolation
        arg2 = self.correction
        return f"{name}({arg1}, {arg2})"

    @abc.abstractmethod
    def solution_from_tcoeffs(self, taylor_coefficients, /, *, num_data_points) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, t, u, marginals, solution: P, /) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: S, /) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_at_terminal_values(self, state: S, /) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, t, *, s0: S, s1: S, output_scale) -> InterpRes[S]:
        raise NotImplementedError

    @abc.abstractmethod
    def case_interpolate(self, t, *, s0: S, s1: S, output_scale) -> InterpRes[S]:
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self, *, t, marginals, posterior: P, posterior_previous: P, t0, t1, output_scale
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def _begin_extrapolation(self, state: S, /, *, dt) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def _complete_extrapolation(
        self, output_extra: S, /, *, output_scale, state_previous: S
    ) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def _begin_correction(self, output_extra: S, /, *, vector_field, p):
        raise NotImplementedError

    @abc.abstractmethod
    def _complete_correction(self, extrapolated: S, /, *, cache_obs):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.extrapolation, self.correction)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (extra, correct) = children
        return cls(extra, correct)

    def promote_output_scale(self, *args, **kwargs):
        return self.extrapolation.promote_output_scale(*args, **kwargs)

    def begin(self, state: S, /, *, dt, parameters, vector_field):
        state = self._begin_extrapolation(state, dt=dt)
        state = self._begin_correction(state, vector_field=vector_field, p=parameters)
        return state

    def complete(
        self, state, state_previous, /, *, vector_field, parameters, output_scale
    ):
        state = self._complete_extrapolation(
            state,
            state_previous=state_previous,
            output_scale=output_scale,
        )
        state = self._complete_correction(
            state, p=parameters, vector_field=vector_field
        )
        return state

    @abc.abstractmethod
    def num_data_points(self, state, /):
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self, state, /):
        raise NotImplementedError
