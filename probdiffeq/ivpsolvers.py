"""Probabilistic IVP solvers."""

from probdiffeq import stats
from probdiffeq.backend import containers, functools, special, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array, Callable, Generic, TypeVar
from probdiffeq.impl import impl

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class _MarkovProcess(containers.NamedTuple):
    tcoeffs: Any
    output_scale: Any
    discretize: Callable

    @property
    def num_derivatives(self):
        return len(self.tcoeffs) - 1


@containers.dataclass
class _InterpRes(Generic[R]):
    step_from: R
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interpolated: R
    """The new 'solution' field.

    At time `t`. This is the interpolation result.
    """

    interp_from: R
    """The new `interp_from` field.

    At time `t`. Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.

    The difference between `interpolated` and `interp_from` emerges in save_at* modes.
    `interpolated` belongs to the just-concluded time interval,
    and `interp_from` belongs to the to-be-started time interval.
    Concretely, this means that `interp_from` has a unit backward model
    and `interpolated` remembers how to step back to the previous target location.
    """


class _PositiveCubatureRule(containers.NamedTuple):
    """Cubature rule with positive weights."""

    points: Array
    weights_sqrtm: Array


def cubature_third_order_spherical(input_shape) -> _PositiveCubatureRule:
    """Third-order spherical cubature integration."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _third_order_spherical_params(d=d)
        return _PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _third_order_spherical_params(d=1)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return _PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _third_order_spherical_params(*, d):
    eye_d = np.eye(d) * np.sqrt(d)
    pts = np.concatenate((eye_d, -1 * eye_d))
    weights_sqrtm = np.ones((2 * d,)) / np.sqrt(2.0 * d)
    return pts, weights_sqrtm


def cubature_unscented_transform(input_shape, r=1.0) -> _PositiveCubatureRule:
    """Unscented transform."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _unscented_transform_params(d=d, r=r)
        return _PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _unscented_transform_params(d=1, r=r)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return _PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _unscented_transform_params(d, *, r):
    eye_d = np.eye(d) * np.sqrt(d + r)
    zeros = np.zeros((1, d))
    pts = np.concatenate((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = np.ones((d,)) / np.sqrt(2.0 * _scale)
    weights_sqrtm2 = np.sqrt(r / _scale)
    weights_sqrtm = np.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


def cubature_gauss_hermite(input_shape, degree=5) -> _PositiveCubatureRule:
    """(Statistician's) Gauss-Hermite cubature.

    The number of cubature points is `prod(input_shape)**degree`.
    """
    assert len(input_shape) == 1
    (dim,) = input_shape

    # Roots of the probabilist/statistician's Hermite polynomials (in Numpy...)
    _roots = special.roots_hermitenorm(n=degree, mu=True)
    pts, weights, sum_of_weights = _roots
    weights = weights / sum_of_weights

    # Transform into jax arrays and take square root of weights
    pts = np.asarray(pts)
    weights_sqrtm = np.sqrt(np.asarray(weights))

    # Build a tensor grid and return class
    tensor_pts = _tensor_points(pts, d=dim)
    tensor_weights_sqrtm = _tensor_weights(weights_sqrtm, d=dim)
    return _PositiveCubatureRule(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


# how does this generalise to an input_shape instead of an input_dimension?
# via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return np.prod_along_axis(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = np.meshgrid(*([x] * d))
    y_mesh = tree_util.tree_map(lambda s: np.reshape(s, (-1,)), x_mesh)
    return np.stack(y_mesh).T


@containers.dataclass
class _ExtraImpl(Generic[T, R, S]):
    """Extrapolation model interface."""

    prior: _MarkovProcess

    initial_condition: Callable
    """Compute an initial condition from a set of Taylor coefficients."""

    init: Callable
    """Initialise a state from a solution."""

    begin: Callable
    """Begin the extrapolation."""

    complete: Callable
    """Complete the extrapolation."""

    extract: Callable
    """Extract a solution from a state."""

    interpolate: Callable
    """Interpolate."""

    interpolate_at_t1: Callable
    """Process the state at checkpoint t=t_n."""


def _extra_impl_precon_smoother(prior: _MarkovProcess, ssm) -> _ExtraImpl:
    def initial_condition():
        rv = ssm.normal.from_tcoeffs(prior.tcoeffs)
        cond = ssm.conditional.identity(len(prior.tcoeffs))
        return stats.MarkovSeq(init=rv, conditional=cond)

    def init(sol: stats.MarkovSeq, /):
        return sol.init, sol.conditional

    def begin(rv, _extra, /, dt):
        cond, (p, p_inv) = prior.discretize(dt)

        rv_p = ssm.normal.preconditioner_apply(rv, p_inv)

        m_p = ssm.stats.mean(rv_p)
        extrapolated_p = ssm.conditional.apply(m_p, cond)

        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(_ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = ssm.conditional.revert(rv_p, (A, noise))
        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)
        cond = ssm.conditional.preconditioner_apply(cond_p, p, p_inv)

        # Gather and return
        return extrapolated, cond

    def extract(hidden_state, extra, /):
        return stats.MarkovSeq(init=hidden_state, conditional=extra)

    def interpolate(state_t0, marginal_t1, *, dt0, dt1, output_scale):
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
        extrapolated_t = _extrapolate(*state_t0, dt0, output_scale)
        extrapolated_t1 = _extrapolate(*extrapolated_t, dt1, output_scale)

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
        solution_at_t = (rv_at_t, extrapolated_t[1])

        # The state at t1 gets a new backward model; it must remember how to
        # get back to t, not to t0.
        solution_at_t1 = (marginal_t1, conditional_t1_to_t)

        return _InterpRes(
            step_from=solution_at_t1,
            interpolated=solution_at_t,
            interp_from=solution_at_t,
        )

    def _extrapolate(state, extra, /, dt, output_scale):
        begun = begin(state, extra, dt=dt)
        return complete(*begun, output_scale=output_scale)

    def interpolate_at_t1(rv, extra, /):
        return _InterpRes((rv, extra), (rv, extra), (rv, extra))

    return _ExtraImpl(
        prior=prior,
        initial_condition=initial_condition,
        init=init,
        begin=begin,
        complete=complete,
        extract=extract,
        interpolate=interpolate,
        interpolate_at_t1=interpolate_at_t1,
    )


def _extra_impl_precon_fixedpoint(prior: _MarkovProcess, *, ssm) -> _ExtraImpl:
    def initial_condition():
        rv = ssm.normal.from_tcoeffs(prior.tcoeffs)
        cond = ssm.conditional.identity(len(prior.tcoeffs))
        return stats.MarkovSeq(init=rv, conditional=cond)

    def init(sol: stats.MarkovSeq, /):
        return sol.init, sol.conditional

    def begin(rv, extra, /, dt):
        cond, (p, p_inv) = prior.discretize(dt)

        rv_p = ssm.normal.preconditioner_apply(rv, p_inv)

        m_ext_p = ssm.stats.mean(rv_p)
        extrapolated_p = ssm.conditional.apply(m_ext_p, cond)

        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p, extra)
        return extrapolated, cache

    def extract(hidden_state, extra, /):
        return stats.MarkovSeq(init=hidden_state, conditional=extra)

    def complete(_rv, extra, /, output_scale):
        cond, (p, p_inv), rv_p, bw0 = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = ssm.conditional.revert(rv_p, (A, noise))
        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)
        cond = ssm.conditional.preconditioner_apply(cond_p, p, p_inv)

        # Merge conditionals
        cond = ssm.conditional.merge(bw0, cond)

        # Gather and return
        return extrapolated, cond

    def interpolate(state_t0, marginal_t1, *, dt0, dt1, output_scale):
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
        extrapolated_t = _extrapolate(*state_t0, dt0, output_scale)
        conditional_id = ssm.conditional.identity(prior.num_derivatives + 1)
        previous_new = (extrapolated_t[0], conditional_id)
        extrapolated_t1 = _extrapolate(*previous_new, dt1, output_scale)

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        return _InterpRes(
            step_from=(marginal_t1, conditional_t1_to_t),
            interpolated=(rv_at_t, extrapolated_t[1]),
            interp_from=previous_new,
        )

    def _extrapolate(state, extra, /, dt, output_scale):
        begun = begin(state, extra, dt=dt)
        return complete(*begun, output_scale=output_scale)

    def interpolate_at_t1(rv, extra, /):
        cond_identity = ssm.conditional.identity(prior.num_derivatives + 1)
        return _InterpRes((rv, cond_identity), (rv, extra), (rv, cond_identity))

    return _ExtraImpl(
        prior=prior,
        init=init,
        initial_condition=initial_condition,
        begin=begin,
        extract=extract,
        complete=complete,
        interpolate=interpolate,
        interpolate_at_t1=interpolate_at_t1,
    )


def _extra_impl_precon_filter(prior: _MarkovProcess, *, ssm) -> _ExtraImpl:
    def init(sol, /):
        return sol, None

    def initial_condition():
        return ssm.normal.from_tcoeffs(prior.tcoeffs)

    def begin(rv, _extra, /, dt):
        cond, (p, p_inv) = prior.discretize(dt)

        rv_p = ssm.normal.preconditioner_apply(rv, p_inv)

        m_ext_p = ssm.stats.mean(rv_p)
        extrapolated_p = ssm.conditional.apply(m_ext_p, cond)

        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def extract(hidden_state, _extra, /):
        return hidden_state

    def complete(_ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p = ssm.conditional.marginalise(rv_p, (A, noise))
        extrapolated = ssm.normal.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        return extrapolated, None

    def interpolate(state_t0, marginal_t1, dt0, dt1, output_scale):
        # todo: by ditching marginal_t1 and dt1, this function _extrapolates
        #  (no *inter*polation happening)
        del dt1

        hidden, extra = state_t0
        hidden, extra = begin(hidden, extra, dt=dt0)
        hidden, extra = complete(hidden, extra, output_scale=output_scale)

        # Consistent state-types in interpolation result.
        interp = (hidden, extra)
        step_from = (marginal_t1, None)
        return _InterpRes(step_from=step_from, interpolated=interp, interp_from=interp)

    def interpolate_at_t1(rv, extra, /):
        return _InterpRes((rv, extra), (rv, extra), (rv, extra))

    return _ExtraImpl(
        prior=prior,
        init=init,
        initial_condition=initial_condition,
        begin=begin,
        extract=extract,
        complete=complete,
        interpolate=interpolate,
        interpolate_at_t1=interpolate_at_t1,
    )


@containers.dataclass
class _Correction:
    """Correction model interface."""

    name: str
    ode_order: int

    init: Callable
    """Initialise the state from the solution."""

    estimate_error: Callable
    """Perform all elements of the correction until the error estimate."""

    complete: Callable
    """Complete what has been left out by `estimate_error`."""

    extract: Callable
    """Extract the solution from the state."""


def correction_ts0(*, ssm, ode_order=1) -> _Correction:
    """Zeroth-order Taylor linearisation."""
    return _correction_constraint_ode_taylor(
        ssm=ssm,
        ode_order=ode_order,
        linearise_fun=ssm.linearise.ode_taylor_0th(ode_order=ode_order),
        name=f"<TS0 with ode_order={ode_order}>",
    )


def correction_ts1(*, ssm, ode_order=1) -> _Correction:
    """First-order Taylor linearisation."""
    return _correction_constraint_ode_taylor(
        ssm=ssm,
        ode_order=ode_order,
        linearise_fun=ssm.linearise.ode_taylor_1st(ode_order=ode_order),
        name=f"<TS1 with ode_order={ode_order}>",
    )


def _correction_constraint_ode_taylor(
    ode_order, linearise_fun, name, *, ssm
) -> _Correction:
    def init(ssv, /):
        obs_like = ssm.prototypes.observed()
        return ssv, obs_like

    def estimate_error(hidden_state, _corr, /, vector_field, t):
        def f_wrapped(s):
            return vector_field(*s, t=t)

        A, b = linearise_fun(f_wrapped, hidden_state.mean)
        observed = ssm.transform.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed, ssm=ssm)
        return error_estimate, observed, (A, b)

    def complete(hidden_state, corr, /):
        A, b = corr
        observed, (_gain, corrected) = ssm.transform.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(ssv, _corr, /):
        return ssv

    return _Correction(
        ode_order=ode_order,
        name=name,
        init=init,
        estimate_error=estimate_error,
        complete=complete,
        extract=extract,
    )


def correction_slr0(cubature_fun=cubature_third_order_spherical) -> _Correction:
    """Zeroth-order statistical linear regression."""
    linearise_fun = impl.linearise.ode_statistical_1st(cubature_fun)
    return _correction_constraint_ode_statistical(
        ode_order=1, linearise_fun=linearise_fun, name=f"<SLR1 with ode_order={1}>"
    )


def correction_slr1(cubature_fun=cubature_third_order_spherical) -> _Correction:
    """First-order statistical linear regression."""
    linearise_fun = impl.linearise.ode_statistical_0th(cubature_fun)
    return _correction_constraint_ode_statistical(
        ode_order=1, linearise_fun=linearise_fun, name=f"<SLR0 with ode_order={1}>"
    )


def _correction_constraint_ode_statistical(
    ode_order, linearise_fun, name
) -> _Correction:
    def init(ssv, /):
        obs_like = impl.prototypes.observed()
        return ssv, obs_like

    def estimate_error(hidden_state, _corr, /, vector_field, t):
        f_wrapped = functools.partial(vector_field, t=t)
        A, b = linearise_fun(f_wrapped, hidden_state)
        observed = impl.conditional.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b, f_wrapped)

    def complete(hidden_state, corr, /):
        # Re-linearise (because the linearisation point changed)
        *_, f_wrapped = corr
        A, b = linearise_fun(f_wrapped, hidden_state)

        # Condition
        observed, (_gain, corrected) = impl.conditional.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(hidden_state, _corr, /):
        return hidden_state

    return _Correction(
        ode_order=ode_order,
        name=name,
        init=init,
        estimate_error=estimate_error,
        complete=complete,
        extract=extract,
    )


def _estimate_error(observed, /, *, ssm):
    # TODO: the functions involved in error estimation are still a bit patchy.
    #  for instance, they assume that they are called in exactly this error estimation
    #  context. Same for prototype_qoi etc.
    zero_data = np.zeros(())
    output_scale = ssm.stats.mahalanobis_norm_relative(zero_data, rv=observed)
    error_estimate_unscaled = np.squeeze(ssm.stats.standard_deviation(observed))
    return output_scale * error_estimate_unscaled


class _StrategyState(containers.NamedTuple):
    t: Any
    hidden: Any
    aux_extra: Any
    aux_corr: Any


@containers.dataclass
class _Strategy:
    """Estimation strategy."""

    name: str

    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int

    prior: _MarkovProcess

    @property
    def num_derivatives(self):
        return self.prior.num_derivatives

    initial_condition: Callable
    """Construct an initial condition from a set of Taylor coefficients."""

    init: Callable
    """Initialise a state from a posterior."""

    begin: Callable
    """Predict the error of an upcoming step."""

    complete: Callable
    """Complete the step after the error has been predicted."""

    extract: Callable
    """Extract the solution from a state."""

    case_interpolate_at_t1: Callable
    """Process the solution in case t=t_n."""

    case_interpolate: Callable

    offgrid_marginals: Callable
    """Compute offgrid_marginals."""


def strategy_smoother(prior, correction: _Correction, /, ssm) -> _Strategy:
    """Construct a smoother."""
    extrapolation_impl = _extra_impl_precon_smoother(prior, ssm=ssm)
    return _strategy(
        extrapolation_impl,
        correction,
        ssm=ssm,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
        name=f"<Smoother with {extrapolation_impl}, {correction}>",
    )


def strategy_fixedpoint(prior, correction: _Correction, /, ssm) -> _Strategy:
    """Construct a fixedpoint-smoother."""
    extrapolation_impl = _extra_impl_precon_fixedpoint(prior, ssm=ssm)
    return _strategy(
        extrapolation_impl,
        correction,
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
        name=f"<Fixedpoint smoother with {extrapolation_impl}, {correction}>",
    )


def strategy_filter(prior, correction: _Correction, /, *, ssm) -> _Strategy:
    """Construct a filter."""
    extrapolation_impl = _extra_impl_precon_filter(prior, ssm=ssm)
    return _strategy(
        extrapolation_impl,
        correction,
        name=f"<Filter with {extrapolation_impl}, {correction}>",
        is_suitable_for_save_at=True,
        is_suitable_for_offgrid_marginals=True,
        is_suitable_for_save_every_step=True,
        ssm=ssm,
    )


def _strategy(
    extrapolation: _ExtraImpl,
    correction: _Correction,
    *,
    name,
    is_suitable_for_save_at,
    is_suitable_for_save_every_step,
    is_suitable_for_offgrid_marginals,
    ssm,
):
    def init(t, posterior, /) -> _StrategyState:
        rv, extra = extrapolation.init(posterior)
        rv, corr = correction.init(rv)
        return _StrategyState(t=t, hidden=rv, aux_extra=extra, aux_corr=corr)

    def initial_condition():
        return extrapolation.initial_condition()

    def begin(state: _StrategyState, /, *, dt, vector_field):
        hidden, extra = extrapolation.begin(state.hidden, state.aux_extra, dt=dt)
        t = state.t + dt
        error, observed, corr = correction.estimate_error(
            hidden, state.aux_corr, vector_field=vector_field, t=t
        )
        state = _StrategyState(t=t, hidden=hidden, aux_extra=extra, aux_corr=corr)
        return error, observed, state

    def complete(state, /, *, output_scale):
        hidden, extra = extrapolation.complete(
            state.hidden, state.aux_extra, output_scale=output_scale
        )
        hidden, corr = correction.complete(hidden, state.aux_corr)
        return _StrategyState(t=state.t, hidden=hidden, aux_extra=extra, aux_corr=corr)

    def extract(state: _StrategyState, /):
        hidden = correction.extract(state.hidden, state.aux_corr)
        sol = extrapolation.extract(hidden, state.aux_extra)
        return state.t, sol

    def case_interpolate_at_t1(state_t1: _StrategyState) -> _InterpRes:
        _tmp = extrapolation.interpolate_at_t1(state_t1.hidden, state_t1.aux_extra)
        step_from, solution, interp_from = (
            _tmp.step_from,
            _tmp.interpolated,
            _tmp.interp_from,
        )

        def _state(x):
            t = state_t1.t
            corr_like = tree_util.tree_map(np.empty_like, state_t1.aux_corr)
            return _StrategyState(t=t, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(step_from)
        solution = _state(solution)
        interp_from = _state(interp_from)
        return _InterpRes(step_from, solution, interp_from)

    def case_interpolate(
        t, *, s0: _StrategyState, s1: _StrategyState, output_scale
    ) -> _InterpRes:
        """Process the solution in case t>t_n."""
        # Interpolate
        interp = extrapolation.interpolate(
            state_t0=(s0.hidden, s0.aux_extra),
            marginal_t1=s1.hidden,
            dt0=t - s0.t,
            dt1=s1.t - t,
            output_scale=output_scale,
        )

        # Turn outputs into valid states

        def _state(t_, x):
            corr_like = tree_util.tree_map(np.empty_like, s0.aux_corr)
            return _StrategyState(t=t_, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(s1.t, interp.step_from)
        interpolated = _state(t, interp.interpolated)
        interp_from = _state(t, interp.interp_from)
        return _InterpRes(
            step_from=step_from, interpolated=interpolated, interp_from=interp_from
        )

    def offgrid_marginals(*, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        if not is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t
        state_t0 = init(t0, posterior_t0)

        interp = extrapolation.interpolate(
            state_t0=(state_t0.hidden, state_t0.aux_extra),
            marginal_t1=marginals_t1,
            dt0=dt0,
            dt1=dt1,
            output_scale=output_scale,
        )

        (marginals, _aux) = interp.interpolated
        u = ssm.stats.qoi(marginals)
        return u, marginals

    return _Strategy(
        name=name,
        init=init,
        initial_condition=initial_condition,
        begin=begin,
        complete=complete,
        extract=extract,
        case_interpolate_at_t1=case_interpolate_at_t1,
        case_interpolate=case_interpolate,
        offgrid_marginals=offgrid_marginals,
        is_suitable_for_save_at=is_suitable_for_save_at,
        is_suitable_for_save_every_step=is_suitable_for_save_every_step,
        prior=extrapolation.prior,
    )


def prior_ibm(tcoeffs, *, ssm_fact: str, output_scale=None) -> _MarkovProcess:
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process."""
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs)

    output_scale_user = output_scale or np.ones_like(ssm.prototypes.output_scale())
    discretize = ssm.conditional.ibm_transitions(len(tcoeffs) - 1, output_scale_user)

    output_scale_calib = np.ones_like(ssm.prototypes.output_scale())
    prior = _MarkovProcess(tcoeffs, output_scale_calib, discretize=discretize)
    return prior, ssm


def prior_ibm_discrete(ts, *, tcoeffs_like, ssm_fact: str, output_scale=None):
    """Compute a time-discretized, multiply-integrated Wiener process."""
    prior, ssm = prior_ibm(tcoeffs_like, output_scale=output_scale, ssm_fact=ssm_fact)
    transitions, (p, p_inv) = functools.vmap(prior.discretize)(np.diff(ts))

    preconditioner_apply_vmap = functools.vmap(ssm.conditional.preconditioner_apply)
    conditionals = preconditioner_apply_vmap(transitions, p, p_inv)

    output_scale = np.ones_like(ssm.prototypes.output_scale())
    init = ssm.normal.standard(len(tcoeffs_like), output_scale=output_scale)
    return stats.MarkovSeq(init, conditionals), ssm


@containers.dataclass
class _Calibration:
    """Calibration implementation."""

    init: Callable
    update: Callable
    extract: Callable


def _calibration_most_recent(*, ssm) -> _Calibration:
    def init(prior):
        return prior

    def update(_state, /, observed):
        return ssm.stats.mahalanobis_norm_relative(0.0, observed)

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def _calibration_none() -> _Calibration:
    def init(prior):
        return prior

    def update(_state, /, observed):
        raise NotImplementedError

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def _calibration_running_mean(*, ssm) -> _Calibration:
    # TODO: if we pass the mahalanobis_relative term to the update() function,
    #  it reduces to a generic stats() module that can also be used for e.g.
    #  marginal likelihoods.
    #  In this case, the _calibration_most_recent() stuff becomes void.

    def init(prior):
        return prior, prior, 0.0

    def update(state, /, observed):
        prior, calibrated, num_data = state

        new_term = ssm.stats.mahalanobis_norm_relative(0.0, observed)
        calibrated = ssm.stats.update_mean(calibrated, new_term, num=num_data)
        return prior, calibrated, num_data + 1.0

    def extract(state, /):
        prior, calibrated, _num_data = state
        return prior, calibrated

    return _Calibration(init=init, update=update, extract=extract)


@containers.dataclass
class _Solver:
    """IVP solver."""

    name: str
    requires_rescaling: bool
    error_contraction_rate: int
    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int

    initial_condition: Callable
    init: Callable
    step: Callable
    extract: Callable
    interpolate: Callable
    interpolate_at_t1: Callable
    offgrid_marginals: Callable


class _SolverState(containers.NamedTuple):
    """Solver state."""

    strategy: Any
    output_scale: Any

    @property
    def t(self):
        return self.strategy.t


def solver_mle(strategy, *, ssm):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """
    name = f"<MLE-solver with {strategy}>"

    def step_mle(state, /, *, dt, vector_field, calibration):
        output_scale_prior, _calibrated = calibration.extract(state.output_scale)
        error, _, state_strategy = strategy.begin(
            state.strategy, dt=dt, vector_field=vector_field
        )

        state_strategy = strategy.complete(
            state_strategy, output_scale=output_scale_prior
        )
        observed = state_strategy.aux_corr

        # Calibrate
        output_scale = calibration.update(state.output_scale, observed=observed)

        # Return
        state = _SolverState(strategy=state_strategy, output_scale=output_scale)
        return dt * error, state

    return _solver_calibrated(
        calibration=_calibration_running_mean(ssm=ssm),
        impl_step=step_mle,
        strategy=strategy,
        name=name,
        requires_rescaling=True,
    )


def solver_dynamic(strategy, *, ssm):
    """Create a solver that calibrates the output scale dynamically."""
    name = f"<Dynamic solver with {strategy}>"

    def step_dynamic(state, /, *, dt, vector_field, calibration):
        error, observed, state_strategy = strategy.begin(
            state.strategy, dt=dt, vector_field=vector_field
        )

        output_scale = calibration.update(state.output_scale, observed=observed)

        prior, _calibrated = calibration.extract(output_scale)
        state_strategy = strategy.complete(state_strategy, output_scale=prior)

        # Return solution
        state = _SolverState(strategy=state_strategy, output_scale=output_scale)
        return dt * error, state

    return _solver_calibrated(
        strategy=strategy,
        calibration=_calibration_most_recent(ssm=ssm),
        name=name,
        impl_step=step_dynamic,
        requires_rescaling=False,
    )


def solver(strategy, /):
    """Create a solver that does not calibrate the output scale automatically."""

    def step(state: _SolverState, *, vector_field, dt, calibration):
        del calibration  # unused

        error, _observed, state_strategy = strategy.begin(
            state.strategy, dt=dt, vector_field=vector_field
        )
        state_strategy = strategy.complete(
            state_strategy, output_scale=state.output_scale
        )
        # Extract and return solution
        state = _SolverState(strategy=state_strategy, output_scale=state.output_scale)
        return dt * error, state

    name = f"<Uncalibrated solver with {strategy}>"
    return _solver_calibrated(
        strategy=strategy,
        calibration=_calibration_none(),
        impl_step=step,
        name=name,
        requires_rescaling=False,
    )


def _solver_calibrated(
    *, calibration, impl_step, strategy, name, requires_rescaling
) -> _Solver:
    def init(t, initial_condition) -> _SolverState:
        posterior, output_scale = initial_condition
        state_strategy = strategy.init(t, posterior)
        calib_state = calibration.init(output_scale)
        return _SolverState(strategy=state_strategy, output_scale=calib_state)

    def step(state: _SolverState, *, vector_field, dt) -> _SolverState:
        return impl_step(
            state, vector_field=vector_field, dt=dt, calibration=calibration
        )

    def extract(state: _SolverState, /):
        t, posterior = strategy.extract(state.strategy)
        _output_scale_prior, output_scale = calibration.extract(state.output_scale)
        return t, (posterior, output_scale)

    def interpolate(
        t, *, interp_from: _SolverState, interp_to: _SolverState
    ) -> _InterpRes:
        output_scale, _ = calibration.extract(interp_to.output_scale)
        interp = strategy.case_interpolate(
            t, s0=interp_from.strategy, s1=interp_to.strategy, output_scale=output_scale
        )
        prev = _SolverState(interp.interp_from, output_scale=interp_from.output_scale)
        sol = _SolverState(interp.interpolated, output_scale=interp_to.output_scale)
        acc = _SolverState(interp.step_from, output_scale=interp_to.output_scale)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)

    def interpolate_at_t1(*, interp_from, interp_to) -> _InterpRes:
        x = strategy.case_interpolate_at_t1(interp_to.strategy)

        prev = _SolverState(x.interp_from, output_scale=interp_from.output_scale)
        sol = _SolverState(x.interpolated, output_scale=interp_to.output_scale)
        acc = _SolverState(x.step_from, output_scale=interp_to.output_scale)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)

    def initial_condition():
        """Construct an initial condition."""
        posterior = strategy.initial_condition()
        return posterior, strategy.prior.output_scale

    return _Solver(
        error_contraction_rate=strategy.num_derivatives + 1,
        is_suitable_for_save_at=strategy.is_suitable_for_save_at,
        is_suitable_for_save_every_step=strategy.is_suitable_for_save_every_step,
        name=name,
        requires_rescaling=requires_rescaling,
        initial_condition=initial_condition,
        init=init,
        step=step,
        extract=extract,
        interpolate=interpolate,
        interpolate_at_t1=interpolate_at_t1,
        offgrid_marginals=strategy.offgrid_marginals,
    )
