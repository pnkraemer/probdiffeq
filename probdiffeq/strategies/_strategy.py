"""Interface for estimation strategies."""

from typing import Generic, TypeVar

import jax

from probdiffeq import _interp

S = TypeVar("S")
"""A type-variable to indicate strategy-state types."""

P = TypeVar("P")
"""A type-variable to indicate strategy-solution ("posterior") types."""


@jax.tree_util.register_pytree_node_class
class Strategy(Generic[S, P]):
    """Inference strategy interface."""

    def __init__(self, extrapolation, correction):
        self.extrapolation = extrapolation
        self.correction = correction

    def __repr__(self):
        name = self.__class__.__name__
        arg1 = self.extrapolation
        arg2 = self.correction
        return f"{name}({arg1}, {arg2})"

    def solution_from_tcoeffs(self, taylor_coefficients, /) -> P:
        raise NotImplementedError

    def init(self, t, solution: P, /) -> S:
        raise NotImplementedError

    def extract(self, state: S, /) -> P:
        raise NotImplementedError

    def case_right_corner(
        self, t, *, s0: S, s1: S, output_scale
    ) -> _interp.InterpRes[S]:
        raise NotImplementedError

    def case_interpolate(
        self, t, *, s0: S, s1: S, output_scale
    ) -> _interp.InterpRes[S]:
        raise NotImplementedError

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

    def begin(self, state: S, /, *, dt, parameters, vector_field):
        raise NotImplementedError

    def complete(self, state, /, *, parameters, vector_field, output_scale):
        raise NotImplementedError
