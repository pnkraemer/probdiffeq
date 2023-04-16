"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq._collections import InterpRes
from probdiffeq.strategies import _strategy


# todo: if we happen to keep this class, make it generic.
class _FiState(NamedTuple):
    """Filtering state."""

    ssv: Any

    # Caches and stuff
    extra: Any
    corr: Any

    # Todo: move this info to ssv
    t: Any
    u: Any

    # todo: is this property a bit hacky?

    @property
    def error_estimate(self):
        return self.corr.error_estimate

    def scale_covariance(self, s, /):
        return _FiState(
            t=self.t,
            u=self.u,
            ssv=self.ssv.scale_covariance(s),
            extra=self.extra.scale_covariance(s),
            corr=self.corr.scale_covariance(s),
        )


S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class FilterDist(_strategy.Posterior[S]):
    """Filtering solution."""

    def __init__(self, rv: S, num_data_points):
        self.rv = rv
        self.num_data_points = num_data_points

    def sample(self, key, *, shape):
        raise NotImplementedError

    def tree_flatten(self):
        children = self.rv, self.num_data_points
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        rv, num_data_points = children
        return cls(rv, num_data_points=num_data_points)


_SolType = Tuple[float, jax.Array, jax.Array, FilterDist]


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy[_FiState, Any]):
    """Filter strategy."""

    def solution_from_tcoeffs(
        self, taylor_coefficients, /, *, num_data_points
    ) -> Tuple[jax.Array, jax.Array, FilterDist]:
        ssv = self.extrapolation.solution_from_tcoeffs_without_reversal(
            taylor_coefficients
        )
        sol = FilterDist(ssv, num_data_points=num_data_points)
        marginals = ssv
        u = taylor_coefficients[0]
        return u, marginals, sol

    def init(self, t, u, _marginals, solution: FilterDist) -> _FiState:
        s, e = self.extrapolation.init_without_reversal(
            solution.rv, solution.num_data_points
        )
        s, c = self.correction.init(s)
        return _FiState(t=t, u=u, ssv=s, extra=e, corr=c)

    def begin(self, state: _FiState, /, *, dt, parameters, vector_field) -> _FiState:
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field, state.t + dt, parameters
        )
        return _FiState(
            t=state.t + dt, u=ssv.extract_qoi(), ssv=ssv, extra=extra, corr=corr
        )

    def complete(self, state, /, *, vector_field, parameters, output_scale):
        ssv, extra = self.extrapolation.complete_without_reversal(
            state.ssv, state.extra, output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, vector_field, state.t, parameters
        )
        return _FiState(t=state.t, u=ssv.extract_qoi(), ssv=ssv, extra=extra, corr=corr)

    def extract(self, post: _FiState, /) -> _SolType:
        t = post.t
        marginals = self.extrapolation.extract_without_reversal(post.ssv, post.extra)
        solution = FilterDist(marginals, self.num_data_points(post))
        u = post.ssv.extract_qoi()
        return t, u, marginals, solution

    def extract_at_terminal_values(self, posterior: _FiState, /) -> _SolType:
        return self.extract(posterior)

    def num_data_points(self, state: _FiState, /):
        return state.ssv.num_data_points

    def observation(self, state, /):
        return state.corr.observed

    def case_right_corner(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> InterpRes[_FiState]:  # s1.t == t
        return InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> InterpRes[_FiState]:
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - s0.t
        ssv, extra = self.extrapolation.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.complete_without_reversal(
            ssv, extra, output_scale
        )
        extrapolated = _FiState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            # Interpolation must be shape- and dtype-stable
            extra=jax.tree_util.tree_map(jnp.empty_like, s1.extra),
            corr=jax.tree_util.tree_map(jnp.empty_like, s1.corr),
        )
        return InterpRes(accepted=s1, solution=extrapolated, previous=extrapolated)

    def offgrid_marginals(
        self,
        *,
        t,
        marginals,
        posterior,
        posterior_previous,
        t0,
        t1,
        output_scale,
    ) -> Tuple[jax.Array, jax.Array]:
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, None, None, posterior),
            s0=self.init(t0, None, None, posterior_previous),
            output_scale=output_scale,
        )
        _, u, marginals, _ = self.extract(sol)
        return u, marginals
