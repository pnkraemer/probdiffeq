"""Estimation strategies for IVP solvers."""

from probdiffeq import _interp
from probdiffeq.backend import abc, containers, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Generic, TypeVar
from probdiffeq.impl import impl
from probdiffeq.solvers import markov

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class _ExtrapolationImpl(abc.ABC, Generic[T, R, S]):
    """Extrapolation model interface."""

    @abc.abstractmethod
    def initial_condition(self, tcoeffs, /) -> T:
        """Compute an initial condition from a set of Taylor coefficients."""
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, solution: T, /) -> tuple[R, S]:
        """Initialise a state from a solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, state: R, aux: S, /, dt) -> tuple[R, S]:
        """Begin the extrapolation."""
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, state: R, aux: S, /, output_scale) -> tuple[R, S]:
        """Complete the extrapolation."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: R, aux: S, /) -> T:
        """Extract a solution from a state."""
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale):
        """Interpolate."""
        raise NotImplementedError

    @abc.abstractmethod
    def right_corner(self, state: R, aux: S, /) -> _interp.InterpRes[tuple[R, S]]:
        """Process the state at checkpoint t=t_n."""
        raise NotImplementedError


class _State(containers.NamedTuple):
    t: Any
    hidden: Any
    aux_extra: Any
    aux_corr: Any


class _Strategy:
    """Estimation strategy."""

    def __init__(
        self,
        extrapolation: _ExtrapolationImpl,
        correction,
        *,
        string_repr,
        is_suitable_for_save_at,
        is_suitable_for_save_every_step,
        is_suitable_for_offgrid_marginals,
    ):
        # Content
        self.extrapolation = extrapolation
        self.correction = correction

        # Some meta-information
        self.string_repr = string_repr
        self.is_suitable_for_save_at = is_suitable_for_save_at
        self.is_suitable_for_save_every_step = is_suitable_for_save_every_step
        self.is_suitable_for_offgrid_marginals = is_suitable_for_offgrid_marginals

    def __repr__(self):
        return self.string_repr

    def initial_condition(self, taylor_coefficients, /):
        """Construct an initial condition from a set of Taylor coefficients."""
        return self.extrapolation.initial_condition(taylor_coefficients)

    def init(self, t, posterior, /) -> _State:
        """Initialise a state from a posterior."""
        rv, extra = self.extrapolation.init(posterior)
        rv, corr = self.correction.init(rv)
        return _State(t=t, hidden=rv, aux_extra=extra, aux_corr=corr)

    def predict_error(self, state: _State, /, *, dt, vector_field):
        """Predict the error of an upcoming step."""
        hidden, extra = self.extrapolation.begin(state.hidden, state.aux_extra, dt=dt)
        t = state.t + dt
        error, observed, corr = self.correction.estimate_error(
            hidden, state.aux_corr, vector_field=vector_field, t=t
        )
        state = _State(t=t, hidden=hidden, aux_extra=extra, aux_corr=corr)
        return error, observed, state

    def complete(self, state, /, *, output_scale):
        """Complete the step after the error has been predicted."""
        hidden, extra = self.extrapolation.complete(
            state.hidden, state.aux_extra, output_scale=output_scale
        )
        hidden, corr = self.correction.complete(hidden, state.aux_corr)
        return _State(t=state.t, hidden=hidden, aux_extra=extra, aux_corr=corr)

    def extract(self, state: _State, /):
        """Extract the solution from a state."""
        hidden = self.correction.extract(state.hidden, state.aux_corr)
        sol = self.extrapolation.extract(hidden, state.aux_extra)
        return state.t, sol

    def case_right_corner(self, state_t1: _State) -> _interp.InterpRes:
        """Process the solution in case t=t_n."""
        _tmp = self.extrapolation.right_corner(state_t1.hidden, state_t1.aux_extra)
        step_from, solution, interp_from = _tmp

        def _state(x):
            t = state_t1.t
            corr_like = tree_util.tree_map(np.empty_like, state_t1.aux_corr)
            return _State(t=t, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(step_from)
        solution = _state(solution)
        interp_from = _state(interp_from)
        return _interp.InterpRes(step_from, solution, interp_from)

    def case_interpolate(
        self, t, *, s0: _State, s1: _State, output_scale
    ) -> _interp.InterpRes[_State]:
        """Process the solution in case t>t_n."""
        # Interpolate
        step_from, solution, interp_from = self.extrapolation.interpolate(
            state_t0=(s0.hidden, s0.aux_extra),
            marginal_t1=s1.hidden,
            dt0=t - s0.t,
            dt1=s1.t - t,
            output_scale=output_scale,
        )

        # Turn outputs into valid states

        def _state(t_, x):
            corr_like = tree_util.tree_map(np.empty_like, s0.aux_corr)
            return _State(t=t_, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(s1.t, step_from)
        solution = _state(t, solution)
        interp_from = _state(t, interp_from)
        return _interp.InterpRes(step_from, solution, interp_from)

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t
        state_t0 = self.init(t0, posterior_t0)

        _acc, (marginals, _aux), _prev = self.extrapolation.interpolate(
            state_t0=(state_t0.hidden, state_t0.aux_extra),
            marginal_t1=marginals_t1,
            dt0=dt0,
            dt1=dt1,
            output_scale=output_scale,
        )

        u = impl.hidden_model.qoi(marginals)
        return u, marginals


def _tree_flatten(strategy):
    children = ()
    aux = (
        # Content
        strategy.extrapolation,
        strategy.correction,
        # Meta-info
        strategy.string_repr,
        strategy.is_suitable_for_offgrid_marginals,
        strategy.is_suitable_for_save_every_step,
        strategy.is_suitable_for_save_at,
    )
    return children, aux


def _tree_unflatten(aux, _children):
    extra, corr, string, suitable_offgrid, suitable_every, suitable_saveat = aux
    return _Strategy(
        extrapolation=extra,
        correction=corr,
        string_repr=string,
        is_suitable_for_save_at=suitable_saveat,
        is_suitable_for_save_every_step=suitable_every,
        is_suitable_for_offgrid_marginals=suitable_offgrid,
    )


tree_util.register_pytree_node(_Strategy, _tree_flatten, _tree_unflatten)


def smoother_adaptive(prior, correction, /) -> _Strategy:
    """Construct a smoother."""
    extrapolation_impl = _PreconSmoother(*prior)
    return _Strategy(
        extrapolation_impl,
        correction,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
        string_repr=f"<Smoother with {extrapolation_impl}, {correction}>",
    )


class _PreconSmoother(_ExtrapolationImpl):
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


def fixedpoint_adaptive(prior, correction, /) -> _Strategy:
    """Construct a fixedpoint-smoother."""
    extrapolation_impl = _PreconFixedPoint(*prior)
    return _Strategy(
        extrapolation_impl,
        correction,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
        string_repr=f"<Fixedpoint smoother with {extrapolation_impl}, {correction}>",
    )


class _PreconFixedPoint(_ExtrapolationImpl):
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


def filter_adaptive(prior, correction, /) -> _Strategy:
    """Construct a filter."""
    extrapolation_impl = _PreconFilter(*prior)
    return _Strategy(
        extrapolation_impl,
        correction,
        string_repr=f"<Filter with {extrapolation_impl}, {correction}>",
        is_suitable_for_save_at=True,
        is_suitable_for_offgrid_marginals=True,
        is_suitable_for_save_every_step=True,
    )


class _PreconFilter(_ExtrapolationImpl):
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
