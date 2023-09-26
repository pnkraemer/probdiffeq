"""Adaptive(/continuous-time) fixedpoint-smoother implementations."""

from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.solvers import markov
from probdiffeq.solvers.strategies import strategy


def fixedpoint_adaptive(prior, correction, /) -> strategy.Strategy:
    """Construct a fixedpoint-smoother."""
    extrapolation_impl = _PreconFixedPoint(*prior)
    return strategy.Strategy(
        extrapolation_impl,
        correction,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
        string_repr=f"<Fixedpoint smoother with {extrapolation_impl}, {correction}>",
    )


class _PreconFixedPoint(strategy.ExtrapolationImpl):
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

    def begin(self, rv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = impl.stats.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_ext_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p, extra)
        return extrapolated, cache

    def complete(self, _rv, extra, /, output_scale):
        cond, (p, p_inv), rv_p, bw0 = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.variable.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = impl.conditional.revert(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = impl.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Merge conditionals
        cond = impl.conditional.merge(bw0, cond)

        # Gather and return
        return extrapolated, cond

    def reset(self, ssv, _extra, /):
        cond = impl.ssm_util.identity_conditional(self.num_derivatives + 1)
        return ssv, cond

    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale):
        """Interpolate.

        A fixed-point smoother interpolates by

        * Extrapolating from t0 to t, which gives the "filtering" marginal
          and the backward transition from t to t0.
        * Extrapolating from t to t1, which gives another "filtering" marginal
          and the backward transition from t1 to t.
        * Applying the t1-to-t backward transition to compute the interpolation result.
          This intermediate result is informed about its "right-hand side" datum.

        The difference to smoother-interpolation is quite subtle:

        * The backward transition of the solution at 't' is merged with that at 't0'.
          The reason is that the backward transition at 't0' knows
          "how to get to the quantity of interest",
          and this is precisely what we want to interpolate.
        * Subsequent interpolations do not continue from the value at 't', but
          from a very similar value where the backward transition
          is replaced with an identity. The reason is that the interpolated solution
          becomes the new quantity of interest, and subsequent interpolations
          need to learn how to get here.
        * Subsequent solver steps do not continue from the value at 't1',
          but the value at 't1' where the backward model is replaced by
          the 't1-to-t' backward model. The reason is similar to the above:
          future steps need to know "how to get back to the quantity of interest",
          which is the interpolated solution.

        These distinctions are precisely why we need three fields
        in every interpolation result:
            the solution,
            the continue-interpolation-from-here,
            and the continue-stepping-from-here.
        All three are different for fixed point smoothers.
        (Really, I try removing one of them monthly and
        then don't understand why tests fail.)
        """
        # Extrapolate from t0 to t, and from t to t1. This yields all building blocks.
        extrapolated_t = self._extrapolate(*state_t0, dt0, output_scale)
        conditional_id = impl.ssm_util.identity_conditional(self.num_derivatives + 1)
        previous_new = (extrapolated_t[0], conditional_id)
        extrapolated_t1 = self._extrapolate(*previous_new, dt1, output_scale)

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = impl.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        return _interp.InterpRes(
            accepted=(marginal_t1, conditional_t1_to_t),
            solution=(rv_at_t, extrapolated_t[1]),
            previous=previous_new,
        )

    def _extrapolate(self, state, extra, /, dt, output_scale):
        begun = self.begin(state, extra, dt=dt)
        return self.complete(*begun, output_scale=output_scale)

    # todo: rename to prepare_future_steps?
    def right_corner(self, rv, extra, /):
        cond_identity = impl.ssm_util.identity_conditional(self.num_derivatives + 1)
        return _interp.InterpRes((rv, cond_identity), (rv, extra), (rv, cond_identity))
