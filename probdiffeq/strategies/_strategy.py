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
    def __init__(self, rv: R, /):
        self.rv = rv

    def tree_flatten(self):
        children = (self.rv,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (rv,) = children
        return cls(rv)

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
    def solution_from_tcoeffs(self, taylor_coefficients, /) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, t, solution: P, /) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: S, /) -> P:
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

    @abc.abstractmethod
    def begin(self, state: S, /, *, dt, parameters, vector_field):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, state, /, *, parameters, vector_field, output_scale):
        raise NotImplementedError
