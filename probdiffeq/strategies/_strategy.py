"""Interface for estimation strategies."""

from typing import Any, Generic, TypeVar

import jax

from probdiffeq import _interp
from probdiffeq.backend import containers
from probdiffeq.strategies import _strategy

S = TypeVar("S")
"""A type-variable to indicate strategy-state types."""

P = TypeVar("P")
"""A type-variable to indicate strategy-solution ("posterior") types."""


class State(containers.NamedTuple):
    t: Any
    ssv: Any
    extra: Any

    corr: Any

    @property
    def u(self):
        return self.ssv.u


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
        # no calibration in __repr__ because it will leave again soon.
        return f"{name}({arg1}, {arg2})"

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        sol = self.extrapolation.solution_from_tcoeffs(taylor_coefficients)
        marginals = sol
        u = taylor_coefficients[0]
        return u, marginals, sol

    def init(self, t, posterior, /) -> _strategy.State:
        ssv, extra = self.extrapolation.init(posterior)
        ssv, corr = self.correction.init(ssv)
        return _strategy.State(t=t, ssv=ssv, extra=extra, corr=corr)

    def begin(self, state: _strategy.State, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t, p=parameters
        )
        return _strategy.State(t=state.t + dt, ssv=ssv, extra=extra, corr=corr)

    def complete(self, state, /, *, output_scale, parameters, vector_field):
        ssv, extra = self.extrapolation.complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, p=parameters, t=state.t, vector_field=vector_field
        )
        return _strategy.State(t=state.t, ssv=ssv, extra=extra, corr=corr)

    def extract(self, state: _strategy.State, /):
        ssv = self.correction.extract(state.ssv, state.corr)
        sol = self.extrapolation.extract(ssv, state.extra)
        return state.t, sol

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
        # todo: they should all be 'aux'?
        children = (self.correction,)
        aux = (self.extrapolation,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*aux, *children)
