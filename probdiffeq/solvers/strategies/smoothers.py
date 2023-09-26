"""Adaptive(/continuous-time) filter implementations."""

from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.solvers import markov
from probdiffeq.solvers.strategies import strategy


def smoother_adaptive(prior, correction, /) -> strategy.Strategy:
    """Construct a smoother."""
    extrapolation_impl = _PreconSmoother(*prior)
    return strategy.Strategy(
        extrapolation_impl,
        correction,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
        string_repr=f"<Smoother with {extrapolation_impl}, {correction}>",
    )


class _PreconSmoother(strategy.ExtrapolationImpl):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def initial_condition(self, tcoeffs, /):
        rv = impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = impl.ssm_util.identity_conditional(len(tcoeffs))
        return markov.MarkovSeq(init=rv, conditional=cond)

    def init(self, sol: markov.MarkovSeq, /):
        return sol.init, sol.conditional

    def extract(self, hidden_state, extra, /):
        return markov.MarkovSeq(init=hidden_state, conditional=extra)

    def begin(self, rv, _extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_p = impl.stats.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.variable.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = impl.conditional.revert(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = impl.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Gather and return
        return extrapolated, cond

    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale):
        """Interpolate.

        A smoother interpolates by_
        * Extrapolating from t0 to t, which gives the "filtering" marginal
          and the backward transition from t to t0.
        * Extrapolating from t to t1, which gives another "filtering" marginal
          and the backward transition from t1 to t.
        * Applying the new t1-to-t backward transition to compute the interpolation.
          This intermediate result is informed about its "right-hand side" datum.

        Subsequent interpolations continue from the value at 't'.
        Subsequent IVP solver steps continue from the value at 't1'.
        """
        # Extrapolate from t0 to t, and from t to t1. This yields all building blocks.
        extrapolated_t = self._extrapolate(*state_t0, dt0, output_scale)
        extrapolated_t1 = self._extrapolate(*extrapolated_t, dt1, output_scale)

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = impl.conditional.marginalise(marginal_t1, conditional_t1_to_t)
        solution_at_t = (rv_at_t, extrapolated_t[1])

        # The state at t1 gets a new backward model; it must remember how to
        # get back to t, not to t0.
        solution_at_t1 = (marginal_t1, conditional_t1_to_t)

        return _interp.InterpRes(
            accepted=solution_at_t1, solution=solution_at_t, previous=solution_at_t
        )

    def _extrapolate(self, state, extra, /, dt, output_scale):
        begun = self.begin(state, extra, dt=dt)
        return self.complete(*begun, output_scale=output_scale)

    def right_corner(self, rv, extra, /):
        return _interp.InterpRes((rv, extra), (rv, extra), (rv, extra))
