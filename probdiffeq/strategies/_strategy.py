"""Interface for estimation strategies."""

import abc
from typing import Generic, TypeVar

import jax

from probdiffeq._collections import InterpRes

S = TypeVar("S")
"""A type-variable to indicate strategy-state types."""

P = TypeVar("P")
"""A type-variable to indicate strategy-solution ("posterior") types."""


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
    def init(self, t, solution: P, /) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: S, /) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_u(self, *, state: S):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_marginals(self, state: P, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_marginals_terminal_values(self, state: P, /):
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
    def sample(self, key, *, posterior: P, shape):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, state: S, /, *, dt) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, output_extra: S, /, *, output_scale, state_previous: S
    ) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def begin_correction(self, output_extra: S, /, *, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, extrapolated: S, /, *, cache_obs):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.extrapolation, self.correction)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (extra, correct) = children
        return cls(extra, correct)

    def init_error_estimate(self):
        return self.extrapolation.init_error_estimate()

    def promote_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.promote_output_scale
        return init_fn(*args, **kwargs)

    def begin(self, state: S, /, *, t, dt, parameters, vector_field):
        # todo (next!): make this return a state-type.
        output_extra = self.begin_extrapolation(state, dt=dt)
        output_corr = self.begin_correction(
            output_extra, vector_field=vector_field, t=t + dt, p=parameters
        )
        return output_extra, output_corr

    def complete(self, output_extra, state, /, *, cache_obs, output_scale):
        # todo: make this operate on state-types.
        extrapolated = self.complete_extrapolation(
            output_extra,
            state_previous=state,
            output_scale=output_scale,
        )
        observed, corrected = self.complete_correction(
            extrapolated, cache_obs=cache_obs
        )
        return observed, corrected

    @abc.abstractmethod
    def num_data_points(self, state, /):
        raise NotImplementedError
