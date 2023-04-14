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

    t: Any
    u: Any
    ssv: Any
    num_data_points: float

    # todo: is this property a bit hacky?

    @property
    def error_estimate(self):
        return self.ssv.error_estimate

    def scale_covariance(self, s, /):
        return _FiState(
            t=self.t,
            u=self.u,
            ssv=self.ssv.scale_covariance(s),
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

    def init(self, t, u, _marginals, solution: FilterDist) -> _FiState:
        x = self.extrapolation.init_without_reversal(solution.rv)
        x = self.correction.init(x)
        return _FiState(
            t=t,
            u=u,
            ssv=x,
            num_data_points=solution.num_data_points,
        )

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

    def extract(self, posterior: _FiState, /) -> _SolType:
        t = posterior.t
        marginals = self.extrapolation.extract_without_reversal(posterior.ssv)
        solution = FilterDist(marginals, posterior.num_data_points)
        u = posterior.ssv.extract_qoi()
        return t, u, marginals, solution

    def extract_at_terminal_values(self, posterior: _FiState, /) -> _SolType:
        return self.extract(posterior)

    def case_right_corner(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> InterpRes[_FiState]:  # s1.t == t
        # s1 = self.init(*self.extract(s1))
        print(jax.tree_util.tree_map(jnp.shape, s1))
        print()
        return InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, t, *, s0: _FiState, s1: _FiState, output_scale
    ) -> InterpRes[_FiState]:
        # s0 = self.init(*self.extract(s0))
        # s1 = self.init(*self.extract(s1))
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - s0.t
        ssv = self.extrapolation.begin(s0.ssv, dt=dt)
        ssv = self.extrapolation.complete_without_reversal(
            ssv,
            state_previous=s0.ssv,
            output_scale=output_scale,
        )
        extrapolated = _FiState(
            t=t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            num_data_points=s0.num_data_points,
        )
        print(jax.tree_util.tree_map(jnp.shape, extrapolated))
        print(jax.tree_util.tree_map(jnp.shape, extrapolated))
        print(jax.tree_util.tree_map(jnp.shape, s1))
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

    def begin(self, state: _FiState, /, *, dt, parameters, vector_field) -> _FiState:
        ssv = self.extrapolation.begin(state.ssv, dt=dt)
        ssv = self.correction.begin(ssv, vector_field, state.t + dt, parameters)
        return _FiState(
            t=state.t + dt,
            u=ssv.extract_qoi(),
            ssv=ssv,
            num_data_points=state.num_data_points,
        )

    def complete(
        self, state, state_previous, /, *, vector_field, parameters, output_scale
    ):
        ssv = self.extrapolation.complete_without_reversal(
            state.ssv,
            state_previous=state_previous.ssv,
            output_scale=output_scale,
        )
        ssv = self.correction.complete(ssv, vector_field, state.t, parameters)
        return _FiState(
            t=state.t,
            u=ssv.extract_qoi(),
            ssv=ssv,
            num_data_points=state.num_data_points + 1,
        )

    def num_data_points(self, state: _FiState, /):
        return state.num_data_points

    def observation(self, state, /):
        return state.ssv.observed_state
