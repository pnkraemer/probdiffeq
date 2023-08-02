"""Interface for estimation strategies."""

from typing import Any, Generic, TypeVar

import jax

from probdiffeq import _interp
from probdiffeq.backend import containers
from probdiffeq.strategies import _strategy

P = TypeVar("P")
"""A type-variable to indicate solution ("posterior") types."""


class State(containers.NamedTuple):
    t: Any
    ssv: Any
    extra: Any

    corr: Any

    @property
    def u(self):
        return self.ssv.u


@jax.tree_util.register_pytree_node_class
class Strategy(Generic[P]):
    """Inference strategy interface."""

    def __init__(
        self,
        extrapolation,
        correction,
        *,
        interpolate_fun,
        string_repr,
        right_corner_fun,
        offgrid_marginals_fun,
        is_suitable_for_save_at,
    ):
        self.extrapolation = extrapolation
        self.correction = correction

        self.is_suitable_for_save_at = is_suitable_for_save_at
        self._string_repr = string_repr
        self._interpolate_fun = interpolate_fun
        self._right_corner_fun = right_corner_fun
        self._offgrid_marginals_fun = offgrid_marginals_fun

    def __repr__(self):
        return self._string_repr

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        return self.extrapolation.solution_from_tcoeffs(taylor_coefficients)

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
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        if self._right_corner_fun is not None:
            return self._right_corner_fun(
                t,
                s0=s0,
                s1=s1,
                output_scale=output_scale,
                extrapolation=self.extrapolation,
            )
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _strategy.State, s1: _strategy.State, output_scale
    ) -> _interp.InterpRes[_strategy.State]:
        return self._interpolate_fun(
            t, output_scale=output_scale, s0=s0, s1=s1, extrapolation=self.extrapolation
        )

    def offgrid_marginals(
        self, *, t, marginals, posterior: P, posterior_previous: P, t0, t1, output_scale
    ):
        if self._offgrid_marginals_fun is None:
            raise NotImplementedError
        return self._offgrid_marginals_fun(
            t,
            marginals=marginals,
            output_scale=output_scale,
            posterior=posterior,
            posterior_previous=posterior_previous,
            t0=t0,
            t1=t1,
            init=self.init,
            interpolate=self.case_interpolate,
            extract=self.extract,
        )

    def tree_flatten(self):
        # todo: they should all be 'aux'?
        children = (self.correction,)
        aux = (
            self.extrapolation,
            self._interpolate_fun,
            self._right_corner_fun,
            self._offgrid_marginals_fun,
            self._string_repr,
            self.is_suitable_for_save_at,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (corr,) = children
        extra, interp, right_corner, offgrid, string, suitable = aux
        return cls(
            extrapolation=extra,
            correction=corr,
            interpolate_fun=interp,
            right_corner_fun=right_corner,
            offgrid_marginals_fun=offgrid,
            string_repr=string,
            is_suitable_for_save_at=suitable,
        )
