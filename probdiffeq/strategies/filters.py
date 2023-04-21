"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import _interp
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

    @property
    def u(self):
        return self.ssv.u

    def scale_covariance(self, s, /):
        ssv = self.ssv.scale_covariance(s)
        corr = self.corr.scale_covariance(s)
        # 'extra' is always None when filtering
        return _FiState(t=self.t, extra=None, ssv=ssv, corr=corr)


S = TypeVar("S")
"""A type-variable to alias appropriate state-space variable types."""


@jax.tree_util.register_pytree_node_class
class FilterDist(_strategy.Posterior[S]):
    """Filtering solution."""

    def sample(self, key, *, shape):
        raise NotImplementedError

    def marginals_at_terminal_values(self):
        marginals = self.rand
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals

    def marginals(self):
        marginals = self.rand
        u = marginals.extract_qoi_from_sample(marginals.mean)
        return u, marginals


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy[_FiState, Any]):
    """Filter strategy."""

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        sol = self.extrapolation.filter.solution_from_tcoeffs(taylor_coefficients)
        sol = FilterDist(sol)
        marginals = sol
        u = taylor_coefficients[0]
        return u, marginals, sol

    def init(self, t, solution, /) -> _FiState:
        ssv, extra = self.extrapolation.filter.init(solution.rand)
        ssv, corr = self.correction.init(ssv)
        return _FiState(t=t, ssv=ssv, extra=extra, corr=corr)

    def extract(self, posterior: _FiState, /):
        t = posterior.t
        ssv = self.correction.extract(posterior.ssv, posterior.corr)
        rv = self.extrapolation.filter.extract(ssv, posterior.extra)

        solution = FilterDist(rv)  # type: ignore
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

        ssv, extra = self.extrapolation.filter.begin(s0.ssv, s0.extra, dt=dt)
        ssv, extra = self.extrapolation.filter.complete(
            ssv, extra, output_scale=output_scale
        )

        corr_like = jax.tree_util.tree_map(jnp.empty_like, s0.corr)
        extrapolated = _FiState(t=t, ssv=ssv, extra=extra, corr=corr_like)
        return _interp.InterpRes(
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
    ) -> Tuple[jax.Array, jax.Array]:
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            s1=self.init(t1, posterior),
            s0=self.init(t0, posterior_previous),
            output_scale=output_scale,
        )
        t, posterior = self.extract(sol)
        return posterior.marginals()

    def begin(self, state: _FiState, /, *, dt, parameters, vector_field):
        ssv, extra = self.extrapolation.filter.begin(state.ssv, state.extra, dt=dt)
        ssv, corr = self.correction.begin(
            ssv, state.corr, vector_field=vector_field, t=state.t + dt, p=parameters
        )
        return _FiState(t=state.t + dt, ssv=ssv, corr=corr, extra=extra)

    def complete(self, state, /, *, output_scale, parameters, vector_field):
        ssv, extra = self.extrapolation.filter.complete(
            state.ssv, state.extra, output_scale=output_scale
        )
        ssv, corr = self.correction.complete(
            ssv, state.corr, p=parameters, t=state.t, vector_field=vector_field
        )
        return _FiState(t=state.t, ssv=ssv, extra=extra, corr=corr)

    def init_error_estimate(self):
        return self.extrapolation.filter.init_error_estimate()

    def promote_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.filter.promote_output_scale
        return init_fn(*args, **kwargs)

    def extract_output_scale(self, *args, **kwargs):
        init_fn = self.extrapolation.filter.extract_output_scale
        return init_fn(*args, **kwargs)
