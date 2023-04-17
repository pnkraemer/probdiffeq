"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple, TypeVar

import jax

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

    t: Any
    u: Any
    extrapolated: Any
    corrected: Any
    num_data_points: float

    def scale_covariance(self, s, /):
        # unexpectedly early call to scale_covariance...
        if self.extrapolated is not None:
            raise ValueError

        return _FiState(
            t=self.t,
            u=self.u,
            extrapolated=None,
            corrected=self.corrected.scale_covariance(s),
            num_data_points=self.num_data_points,
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

    def init(self, t, u, _marginals, solution) -> _FiState:
        return _FiState(
            t=t,
            u=u,
            extrapolated=None,
            corrected=solution.rv,
            num_data_points=solution.num_data_points,
        )

    def solution_from_tcoeffs(
        self, taylor_coefficients, /, *, num_data_points
    ) -> Tuple[jax.Array, jax.Array, FilterDist]:
        ssv = self.extrapolation.solution_from_tcoeffs(taylor_coefficients)
        sol = FilterDist(ssv, num_data_points=num_data_points)
        marginals = ssv
        u = taylor_coefficients[0]
        return u, marginals, sol

    def extract(self, posterior: _FiState, /) -> _SolType:
        t = posterior.t
        solution = FilterDist(posterior.corrected, posterior.num_data_points)
        marginals = solution.rv
        u = marginals.extract_qoi()
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

        output_extra = self.extrapolation.begin(s0.corrected, dt=dt)
        extrapolated = self.extrapolation.complete_without_reversal(
            output_extra,
            s0=s0.corrected,
            output_scale=output_scale,
        )
        extrapolated = _FiState(
            t=t,
            u=extrapolated.extract_qoi(),
            extrapolated=None,
            corrected=extrapolated,
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
        extrapolated = self.extrapolation.begin(state.corrected, dt=dt)
        output_corr = self.correction.begin(
            extrapolated, vector_field=vector_field, t=t + dt, p=parameters
        )

        extrapolated = _FiState(
            t=t + dt,
            u=None,
            corrected=None,
            extrapolated=extrapolated,
            num_data_points=state.num_data_points,
        )
        return extrapolated, output_corr

    def complete(self, output_extra, state, /, *, cache_obs, output_scale):
        extrapolated = self.extrapolation.complete_without_reversal(
            output_extra.extrapolated,
            s0=state.corrected,
            output_scale=output_scale,
        )

        obs, corr = self.correction.complete(extrapolated=extrapolated, cache=cache_obs)
        corr = _FiState(
            t=output_extra.t,
            u=corr.extract_qoi(),
            extrapolated=None,
            corrected=corr,
            num_data_points=output_extra.num_data_points + 1,
        )
        return obs, corr

    def num_data_points(self, state: _FiState, /):
        return state.num_data_points
