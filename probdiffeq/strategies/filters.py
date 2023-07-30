"""Forward-only estimation: filtering."""
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.backend import containers
from probdiffeq.strategies import _strategy


def filter(*impl):
    """Create a filter strategy."""
    extra, corr, calib = impl
    return _Filter(extra.filter, corr), calib


class _FiState(containers.NamedTuple):
    """Filtering state."""

    ssv: Any
    extra: Any
    corr: Any

    # Todo: move those to `ssv`?
    t: Any

    @property
    def u(self):
        return self.ssv.u


S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class _Filter(_strategy.Strategy[_FiState, Any]):
    """Filter strategy."""

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        sol = self.extrapolation.solution_from_tcoeffs(taylor_coefficients)
        marginals = sol
        u = taylor_coefficients[0]
        return u, marginals, sol

    def init(self, t, solution, /) -> _FiState:
        ssv, extra = self.extrapolation.init(solution)
        ssv, corr = self.correction.init(ssv)
        return _FiState(t=t, ssv=ssv, extra=extra, corr=corr)

    def extract(self, posterior: _FiState, /):
        t = posterior.t
        ssv = self.correction.extract(posterior.ssv, posterior.corr)
        solution = self.extrapolation.extract(ssv, posterior.extra)
        return t, solution

    def case_right_corner(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> _interp.InterpRes[_FiState]:  # s1.t == t
        return _interp.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> _interp.InterpRes[_FiState]:
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - s0.t

        ssv, extra = self.extrapolation.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.complete(ssv, extra, output_scale=output_scale)

        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        extrapolated = _FiState(t=t, ssv=ssv, extra=extra, corr=corr_like)
        return _interp.InterpRes(
            accepted=s1, solution=extrapolated, previous=extrapolated
        )

    def offgrid_marginals(
        self, *, t, marginals, posterior, posterior_previous, t0, t1, output_scale
    ):
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, posterior),
            s0=self.init(t0, posterior_previous),
            output_scale=output_scale,
        )
        t, posterior = self.extract(sol)
        u = posterior.extract_qoi_from_sample(posterior.mean)
        return u, posterior

    def begin(self, state: _FiState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t + dt, p=parameters
        )
        return _FiState(t=state.t + dt, ssv=ssv, corr=corr, extra=extra)

    def complete(self, state, /, *, output_scale, parameters, vector_field):
        ssv, extra = self.extrapolation.complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, p=parameters, t=state.t, vector_field=vector_field
        )
        return _FiState(t=state.t, ssv=ssv, extra=extra, corr=corr)
