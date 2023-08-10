"""Interface for estimation strategies."""

from typing import Generic, TypeVar

import jax

from probdiffeq import _interp
from probdiffeq.solvers.strategies import _common

P = TypeVar("P")
"""A type-variable to indicate solution ("posterior") types."""


@jax.tree_util.register_pytree_node_class
class Strategy(Generic[P]):
    """Inference strategy interface."""

    def __init__(
        self,
        extrapolation,
        correction,
        *,
        string_repr,
        is_suitable_for_save_at,
        impl_interpolate,
        impl_right_corner,  # use "default" for default-behaviour
        impl_offgrid_marginals,  # use None if not available
    ):
        # Content
        self.extrapolation = extrapolation
        self.correction = correction

        # Some meta-information
        self.string_repr = string_repr
        self.is_suitable_for_save_at = is_suitable_for_save_at

        # Implementations of functionality that varies
        self.impl_interpolate = impl_interpolate
        self.impl_right_corner = impl_right_corner
        self.impl_offgrid_marginals = impl_offgrid_marginals

    def __repr__(self):
        return self.string_repr

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        return self.extrapolation.solution_from_tcoeffs(taylor_coefficients)

    def init(self, t, posterior, /) -> _common.State:
        rv, extra = self.extrapolation.init(posterior)
        rv, corr = self.correction.init(rv)
        return _common.State(t=t, hidden=rv, aux_extra=extra, aux_corr=corr)

    def predict_error(self, state: _common.State, /, *, dt, parameters, vector_field):
        hidden, extra = self.extrapolation.begin(state.hidden, state.aux_extra, dt=dt)
        error, observed, corr = self.correction.estimate_error(
            hidden, state.aux_corr, vector_field=vector_field, t=state.t, p=parameters
        )
        t = state.t + dt
        state = _common.State(t=t, hidden=hidden, aux_extra=extra, aux_corr=corr)
        return error, observed, state

    def complete(self, state, /, *, output_scale):
        hidden, extra = self.extrapolation.complete(
            state.hidden, state.aux_extra, output_scale=output_scale
        )
        hidden, corr = self.correction.complete(hidden, state.aux_corr)
        return _common.State(t=state.t, hidden=hidden, aux_extra=extra, aux_corr=corr)

    def extract(self, state: _common.State, /):
        hidden = self.correction.extract(state.hidden, state.aux_corr)
        sol = self.extrapolation.extract(hidden, state.aux_extra)
        return state.t, sol

    def case_right_corner(self, state_at_t1: _common.State) -> _interp.InterpRes:
        # If specific choice is provided, use that.
        if self.impl_right_corner != "default":
            return self.impl_right_corner(state_at_t1, extrapolation=self.extrapolation)

        # Otherwise, apply default behaviour.
        s1 = state_at_t1
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _common.State, s1: _common.State, output_scale
    ) -> _interp.InterpRes[_common.State]:
        return self.impl_interpolate(
            t, output_scale=output_scale, s0=s0, s1=s1, extrapolation=self.extrapolation
        )

    def offgrid_marginals(
        self, *, t, marginals, posterior: P, posterior_previous: P, t0, t1, output_scale
    ):
        # If implementation is not provided, then offgrid_marginals is impossible.
        if self.impl_offgrid_marginals is None:
            raise NotImplementedError

        return self.impl_offgrid_marginals(
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
            # Content
            self.extrapolation,
            # Meta-info
            self.string_repr,
            self.is_suitable_for_save_at,
            # Implementations
            self.impl_interpolate,
            self.impl_right_corner,
            self.impl_offgrid_marginals,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (corr,) = children
        extra, string, suitable, interp, right_corner, offgrid = aux
        return cls(
            extrapolation=extra,
            correction=corr,
            string_repr=string,
            is_suitable_for_save_at=suitable,
            impl_interpolate=interp,
            impl_right_corner=right_corner,
            impl_offgrid_marginals=offgrid,
        )
