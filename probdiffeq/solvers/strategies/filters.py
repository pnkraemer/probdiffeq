"""Adaptive(/continuous-time) filter implementations."""

from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import strategy


def filter_adaptive(prior, correction, /) -> strategy.Strategy:
    """Construct a filter."""
    extrapolation_impl = _PreconFilter(*prior)
    return strategy.Strategy(
        extrapolation_impl,
        correction,
        string_repr=f"<Filter with {extrapolation_impl}, {correction}>",
        is_suitable_for_save_at=True,
        is_suitable_for_offgrid_marginals=True,
        is_suitable_for_save_every_step=True,
    )


class _PreconFilter(strategy.ExtrapolationImpl):
    def __init__(self, discretise, num_derivatives):
        # todo: move sol_from_tcoeffs out of this module
        #  (and then we can ditch self.num_derivatives)
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def initial_condition(self, tcoeffs, /):
        return impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)

    def init(self, sol, /):
        return sol, None

    def extract(self, hidden_state, _extra, /):
        return hidden_state

    def begin(self, rv, _extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = impl.stats.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_ext_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.variable.rescale_cholesky(noise, output_scale)
        extrapolated_p = impl.conditional.marginalise(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        return extrapolated, None

    def interpolate(self, state_t0, marginal_t1, dt0, dt1, output_scale):
        # todo: by ditching marginal_t1 and dt1, this function _extrapolates
        #  (no *inter*polation happening)
        del dt1

        hidden, extra = state_t0
        hidden, extra = self.begin(hidden, extra, dt=dt0)
        hidden, extra = self.complete(hidden, extra, output_scale=output_scale)

        # Consistent state-types in interpolation result.
        interp = (hidden, extra)
        step_from = (marginal_t1, None)
        return _interp.InterpRes(accepted=step_from, solution=interp, previous=interp)

    def right_corner(self, rv, extra, /):
        return _interp.InterpRes((rv, extra), (rv, extra), (rv, extra))
