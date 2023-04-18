"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq._collections import InterpRes
from probdiffeq.strategies import _strategy


# this class is NOT redundant.
# next, add "t" into this solution (and into MarkovSequence)
# this will simplify a million functions in this code base
# and is the next step en route to x=extract(init(x)) for solvers, strategies, etc.
# more specifically, init(tcoeffs) becomes
# init_posterior_from_tcoeffs(t, tcoeffs)
#  which allows the solver (!) to satisfy x = extract(init(x)). Then,
#  the strategy can be made to obey this pattern next.
# todo: if we happen to keep this class, make it generic.
class _FiState(NamedTuple):
    """Filtering state."""

    ssv: Any
    extra: Any
    corr: Any

    # Todo: move those to `ssv`
    t: Any
    u: Any
    num_data_points: float

    def scale_covariance(self, s, /):
        return _FiState(
            t=self.t,
            u=self.u,
            extra=None,
            ssv=self.ssv.scale_covariance(s),
            corr=self.corr.scale_covariance(s),
            num_data_points=self.num_data_points,
        )


S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class FilterDist(_strategy.Posterior[S]):
    """Filtering solution."""

    def sample(self, key, *, shape):
        raise NotImplementedError


_SolType = Tuple[float, jax.Array, jax.Array, FilterDist]


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy[_FiState, Any]):
    """Filter strategy."""

    def init(self, t, u, _marginals, solution) -> _FiState:
        ssv, extra = self.extrapolation.filter_init(solution.rv)
        ssv, corr = self.correction.init(ssv)
        return _FiState(
            t=t,
            u=u,
            ssv=ssv,
            extra=extra,
            corr=corr,
            num_data_points=solution.num_data_points,
        )

    def solution_from_tcoeffs(
        self, taylor_coefficients, /, *, num_data_points
    ) -> Tuple[jax.Array, jax.Array, FilterDist]:
        sol = self.extrapolation.solution_from_tcoeffs(taylor_coefficients)
        sol = FilterDist(sol, num_data_points=num_data_points)
        marginals = ssv
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, posterior: _FiState, /) -> _SolType:
        t = posterior.t
        ssv = self.correction.extract(posterior.ssv, posterior.corr)
        rv = self.extrapolation.filter_extract(ssv, posterior.extra)

        solution = FilterDist(rv, posterior.num_data_points)
        marginals = rv
        u = posterior.u
        return t, u, marginals, solution

    def extract_at_terminal_values(self, posterior: _FiState, /) -> _SolType:
        return self.extract(posterior)

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
            ssv, extra, output_scale=output_scale
        )

        extrapolated = _FiState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=jax.tree_util.tree_map(jnp.zeros_like, s0.corr),
            num_data_points=s0.num_data_points,
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

    def begin(self, state: _FiState, /, *, t, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=t + dt, p=parameters
        )
        return _FiState(
            t=t + dt,
            u=ssv.extract_qoi(),
            ssv=ssv,
            corr=corr,
            extra=extra,
            num_data_points=state.num_data_points,
        )

    def complete(self, state, /, *, output_scale, parameters, vector_field):
        ssv, extra = self.extrapolation.complete_without_reversal(
            state.ssv, state.extra, output_scale=output_scale
        )

        ssv, corr = self.correction.complete(
            ssv, state.corr, p=parameters, vector_field=vector_field
        )
        return _FiState(
            t=state.t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            extra=extra,
            corr=corr,
            num_data_points=state.num_data_points + 1,
        )

    def num_data_points(self, state: _FiState, /):
        return state.num_data_points
