"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple

import jax

from probdiffeq import _collections
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

    extrapolated: Any
    corrected: Any
    num_data_points: float

    def scale_covariance(self, s, /):
        return _FiState(self.ssv.scale_covariance(s), self.num_data_points)


class FiSolution(NamedTuple):
    """Filtering solution."""

    rv: Any

    # todo: make a similar field in MarkovSequence
    num_data_points: float


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy[_FiState, Any]):
    """Filter strategy."""

    def init(self, sol: FiSolution, /) -> _FiState:
        return _FiState(
            extrapolated=None,
            corrected=sol.rv,
            num_data_points=sol.num_data_points,
        )

    def solution_from_tcoeffs(
        self, taylor_coefficients, *, num_data_points
    ) -> FiSolution:
        ssv = self.implementation.extrapolation.init_state_space_var(
            taylor_coefficients=taylor_coefficients
        )
        return FiSolution(ssv, num_data_points=num_data_points)

    def extract(self, posterior: _FiState, /) -> FiSolution:
        return FiSolution(posterior.corrected, posterior.num_data_points)

    # todo: make interpolation result into a named-tuple.
    #  it is too confusing what those three posteriors mean.
    def case_right_corner(
        self, *, s0: _FiState, s1: _FiState, t, t0, t1, output_scale
    ) -> _collections.InterpRes[_FiState]:  # s1.t == t
        return _collections.InterpRes(accepted=s1, solution=s1, previous=s1)

    def case_interpolate(
        self, *, s0: _FiState, s1: _FiState, t0, t, t1, output_scale
    ) -> _collections.InterpRes[_FiState]:
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - t0
        output_extra = self.begin_extrapolation(s0, dt=dt)
        extrapolated = self.complete_extrapolation(
            output_extra,
            state_previous=s0,
            output_scale=output_scale,
        )
        # we need to move the extrapolation to the 'correction' field.
        extrapolated = _FiState(
            extrapolated=None,
            corrected=extrapolated.extrapolated,
            num_data_points=extrapolated.num_data_points,
        )
        return _collections.InterpRes(
            accepted=s1, solution=extrapolated, previous=extrapolated
        )

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
    ) -> Tuple[jax.Array, _FiState]:
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            s1=_FiState(posterior),
            s0=_FiState(posterior_previous),
            t0=t0,
            t1=t1,
            output_scale=output_scale,
        )
        u = self.extract_u(sol)
        return u, sol

    def sample(self, key, *, posterior: _FiState, shape):
        raise NotImplementedError

    def extract_marginals(self, sol: FiSolution, /):
        return sol.rv

    def extract_marginals_terminal_values(self, sol: FiSolution, /):
        return sol.rv

    def extract_u(self, *, state: _FiState):
        # todo: should this point to extrapolated or corrected?
        return state.corrected.extract_qoi()

    def begin_extrapolation(self, posterior: _FiState, /, *, dt) -> _FiState:
        extrapolate = self.implementation.extrapolation.begin_extrapolation
        extrapolated = extrapolate(posterior.corrected, dt=dt)
        return _FiState(
            extrapolated=extrapolated,
            corrected=None,
            num_data_points=posterior.num_data_points,
        )

    # todo: make "output_extra" positional only. Then rename this mess.
    def begin_correction(
        self, output_extra: _FiState, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, Any]:
        return self.implementation.correction.begin_correction(
            output_extra.extrapolated, vector_field=vector_field, t=t, p=p
        )

    def complete_extrapolation(
        self,
        output_extra: _FiState,
        /,
        *,
        output_scale,
        state_previous: _FiState,
    ) -> _FiState:
        extra = self.implementation.extrapolation
        extrapolate_fn = extra.complete_extrapolation_without_reversal
        # todo: extrapolation needs a serious signature-variable-renaming...
        ssv = extrapolate_fn(
            output_extra.extrapolated,
            s0=state_previous.corrected,
            output_scale=output_scale,
        )
        return _FiState(
            extrapolated=ssv,
            corrected=None,
            num_data_points=output_extra.num_data_points,
        )

    # todo: more type-stability in corrections!
    def complete_correction(
        self, extrapolated: _FiState, /, *, cache_obs
    ) -> Tuple[Any, Tuple[_FiState, Any]]:
        obs, (corr, gain) = self.implementation.correction.complete_correction(
            extrapolated=extrapolated.extrapolated, cache=cache_obs
        )
        corr = _FiState(
            extrapolated=None,
            corrected=corr,
            num_data_points=extrapolated.num_data_points + 1,
        )
        return obs, (corr, gain)

    def num_data_points(self, state: _FiState, /):
        return state.num_data_points
