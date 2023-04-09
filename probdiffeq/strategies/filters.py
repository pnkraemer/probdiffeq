"""Forward-only estimation: filtering."""
from typing import Any, NamedTuple, Tuple

import jax

from probdiffeq.strategies import _strategy


# this class is NOT redundant.
# next, add "t" into this solution (and into MarkovSequence)
# this will simplify a million functions in this code base
# and is the next step en route to x=extract(init(x)) for solvers, strategies, etc.
# more specifically, init_posterior(tcoeffs) becomes
# init_posterior_from_tcoeffs(t, tcoeffs)
#  which allows the solver (!) to satisfy x = extract(init(x)). Then,
#  the strategy can be made to obey this pattern next.
# todo: if we happen to keep this class, make it generic.
class _FilterSol(NamedTuple):
    """Filtering solution."""

    ssv: Any

    def scale_covariance(self, s, /):
        return _FilterSol(self.ssv.scale_covariance(s))

    def extract_qoi(self):
        return self.ssv.extract_qoi()


@jax.tree_util.register_pytree_node_class
class Filter(_strategy.Strategy[_FilterSol]):
    """Filter strategy."""

    def init_posterior(self, *, taylor_coefficients) -> _FilterSol:
        ssv = self.implementation.extrapolation.init_state_space_var(
            taylor_coefficients=taylor_coefficients
        )
        return _FilterSol(ssv)

    # todo: make interpolation result into a named-tuple.
    #  it is too confusing what those three posteriors mean.
    def case_right_corner(
        self, *, p0: _FilterSol, p1: _FilterSol, t, t0, t1, output_scale
    ) -> Tuple[_FilterSol, _FilterSol, _FilterSol]:  # s1.t == t
        return p1, p1, p1

    def case_interpolate(
        self, *, p0: _FilterSol, rv1, t0, t, t1, output_scale
    ) -> Tuple[_FilterSol, _FilterSol, _FilterSol]:
        # A filter interpolates by extrapolating from the previous time-point
        # to the in-between variable. That's it.
        dt = t - t0
        linearisation_pt = self.begin_extrapolation(posterior=p0, dt=dt)
        extrapolated = self.complete_extrapolation(
            linearisation_pt,
            posterior_previous=p0,
            output_scale=output_scale,
        )
        return _FilterSol(rv1), extrapolated, extrapolated

    def offgrid_marginals(
        self, *, t, marginals, posterior_previous: _FilterSol, t0, t1, output_scale
    ) -> Tuple[jax.Array, _FilterSol]:
        _acc, sol, _prev = self.case_interpolate(
            t=t,
            rv1=marginals,
            p0=posterior_previous,
            t0=t0,
            t1=t1,
            output_scale=output_scale,
        )
        u = self.extract_u_from_posterior(posterior=sol)
        return u, sol

    def sample(self, key, *, posterior: _FilterSol, shape):
        raise NotImplementedError

    def marginals(self, *, posterior: _FilterSol):
        return posterior.ssv

    def marginals_terminal_value(self, *, posterior: _FilterSol):
        return posterior.ssv

    def extract_u_from_posterior(self, posterior: _FilterSol):
        return posterior.ssv.extract_qoi()

    def begin_extrapolation(self, *, posterior: _FilterSol, dt) -> _FilterSol:
        extrapolate = self.implementation.extrapolation.begin_extrapolation
        ssv = extrapolate(posterior.ssv, dt=dt)
        return _FilterSol(ssv)

    # todo: make "linearisation_pt" positional only. Then rename this mess.
    def begin_correction(
        self, linearisation_pt: _FilterSol, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, Any]:
        ssv = linearisation_pt.ssv
        return self.implementation.correction.begin_correction(
            ssv, vector_field=vector_field, t=t, p=p
        )

    def complete_extrapolation(
        self,
        linearisation_pt: _FilterSol,
        *,
        output_scale,
        posterior_previous: _FilterSol,
    ) -> _FilterSol:
        extra = self.implementation.extrapolation
        extrapolate_fn = extra.complete_extrapolation_without_reversal
        # todo: extrapolation needs a serious signature-variable-renaming...
        ssv = extrapolate_fn(
            linearisation_pt=linearisation_pt.ssv,
            p0=posterior_previous.ssv,
            output_scale=output_scale,
        )
        return _FilterSol(ssv)

    # todo: more type-stability in corrections!
    def complete_correction(
        self, *, extrapolated: _FilterSol, cache_obs
    ) -> Tuple[Any, Tuple[_FilterSol, Any]]:
        obs, (corr, gain) = self.implementation.correction.complete_correction(
            extrapolated=extrapolated.ssv, cache=cache_obs
        )
        corr = _FilterSol(corr)
        return obs, (corr, gain)
