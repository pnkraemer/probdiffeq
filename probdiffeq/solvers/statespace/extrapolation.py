"""Extrapolation model interfaces."""

import abc

import jax
import jax.numpy as jnp

from probdiffeq import _interp
from probdiffeq.impl import impl
from probdiffeq.solvers import markov


class Extrapolation(abc.ABC):
    """Extrapolation model interface."""

    @abc.abstractmethod
    def solution_from_tcoeffs(self, taylor_coefficients, /):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, sol, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, ssv, extra, /, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, extra, /, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, extra, /):
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def right_corner(self, rv, extra, /):
        raise NotImplementedError


# At the point of choosing the recipe
# (aka selecting the desired state-space model factorisation),
# it is too early to know whether we solve forward-in-time only (aka filtering)
# or require a dense, or fixed-point solution. Therefore, the recipes return
# extrapolation *factories* instead of extrapolation models.
class ExtrapolationFactory(abc.ABC):
    @abc.abstractmethod
    def string_repr(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def dense(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def save_at(self) -> Extrapolation:
        raise NotImplementedError


class PreconFilter(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        # todo: move sol_from_tcoeffs out of this module
        #  (and then we can ditch self.num_derivatives)
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def solution_from_tcoeffs(self, tcoeffs, /):
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


class PreconSmoother(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = impl.ssm_util.identity_conditional(len(tcoeffs))
        return markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: markov.MarkovSeqRev, /):
        return sol.init, sol.conditional

    def extract(self, hidden_state, extra, /):
        return markov.MarkovSeqRev(init=hidden_state, conditional=extra)

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


class PreconFixedPoint(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = impl.ssm_util.identity_conditional(len(tcoeffs))
        return markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: markov.MarkovSeqRev, /):
        return sol.init, sol.conditional

    def extract(self, hidden_state, extra, /):
        return markov.MarkovSeqRev(init=hidden_state, conditional=extra)

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


class IBMExtrapolationFactory(ExtrapolationFactory):
    def __init__(self, args):
        self.args = args

    def string_repr(self):
        num_derivatives = self.forward().num_derivatives
        return f"<IBM with num_derivatives={num_derivatives}>"

    def forward(self):
        return PreconFilter(*self.args)

    def dense(self):
        return PreconSmoother(*self.args)

    def save_at(self):
        return PreconFixedPoint(*self.args)


def ibm_adaptive(num_derivatives) -> IBMExtrapolationFactory:
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return IBMExtrapolationFactory(args=(discretise, num_derivatives))


def ibm_discretise_fwd(dts, /, *, num_derivatives):
    discretise = impl.ssm_util.ibm_transitions(num_derivatives)
    return jax.vmap(discretise)(dts)


def unit_markov_sequence(num_derivatives):
    cond = impl.ssm_util.identity_conditional(num_derivatives + 1)
    init = impl.ssm_util.standard_normal(num_derivatives + 1, 1.0)
    return markov.MarkovSeqRev(init=init, conditional=cond)
