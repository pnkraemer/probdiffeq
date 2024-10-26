"""Probabilistic IVP solvers."""

from probdiffeq import stats
from probdiffeq.backend import containers, functools, special, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array, Callable, Generic, TypeVar
from probdiffeq.impl import impl

R = TypeVar("R")


class _MarkovProcess(containers.NamedTuple):
    tcoeffs: Any
    output_scale: Any
    discretize: Callable

    @property
    def num_derivatives(self):
        return len(self.tcoeffs) - 1


def prior_ibm(tcoeffs, *, ssm_fact: str, output_scale=None):
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process."""
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs)

    output_scale_user = output_scale or np.ones_like(ssm.prototypes.output_scale())
    discretize = ssm.conditional.ibm_transitions(output_scale=output_scale_user)

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
class _ExtraImpl:
    """Extrapolation model interface."""

    name: str
    ssm: Any

    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int
    is_suitable_for_offgrid_marginals: int

    def initial_condition(self, *, prior):
        """Compute an initial condition from a set of Taylor coefficients."""
        raise NotImplementedError

    def init(self, sol: stats.MarkovSeq, /):
        """Initialise a state from a solution."""
        raise NotImplementedError

    def begin(self, rv, _extra, /, *, prior_discretized):
        """Begin the extrapolation."""
        raise NotImplementedError

    def complete(self, _ssv, extra, /, output_scale):
        """Complete the extrapolation."""
        raise NotImplementedError

    def extract(self, hidden_state, extra, /):
        """Extract a solution from a state."""
        raise NotImplementedError

    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale, prior):
        """Interpolate."""
        raise NotImplementedError

    def interpolate_at_t1(self, rv, extra, /, *, prior):
        """Process the state at checkpoint t=t_n."""
        raise NotImplementedError


@containers.dataclass
class _ExtraImplSmoother(_ExtraImpl):
    def initial_condition(self, *, prior):
        rv = self.ssm.normal.from_tcoeffs(prior.tcoeffs)
        cond = self.ssm.conditional.identity(len(prior.tcoeffs))
        return stats.MarkovSeq(init=rv, conditional=cond)

    def init(self, sol: stats.MarkovSeq, /):
        return sol.init, sol.conditional

    def begin(self, rv, _extra, /, *, prior_discretized):
        cond, (p, p_inv) = prior_discretized

        rv_p = self.ssm.normal.preconditioner_apply(rv, p_inv)

        m_p = self.ssm.stats.mean(rv_p)
        extrapolated_p = self.ssm.conditional.apply(m_p, cond)

        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = self.ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = self.ssm.conditional.revert(rv_p, (A, noise))
        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)
        cond = self.ssm.conditional.preconditioner_apply(cond_p, p, p_inv)

        # Gather and return
        return extrapolated, cond

    def extract(self, hidden_state, extra, /):
        return stats.MarkovSeq(init=hidden_state, conditional=extra)

    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale, prior):
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
        prior0 = prior.discretize(dt0)
        extrapolated_t = self._extrapolate(
            *state_t0, output_scale=output_scale, prior_discretized=prior0
        )

        prior1 = prior.discretize(dt1)
        extrapolated_t1 = self._extrapolate(
            *extrapolated_t, output_scale=output_scale, prior_discretized=prior1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
        solution_at_t = (rv_at_t, extrapolated_t[1])

        # The state at t1 gets a new backward model; it must remember how to
        # get back to t, not to t0.
        solution_at_t1 = (marginal_t1, conditional_t1_to_t)

        return _InterpRes(
            step_from=solution_at_t1,
            interpolated=solution_at_t,
            interp_from=solution_at_t,
        )

    def _extrapolate(self, state, extra, /, *, output_scale, prior_discretized):
        state, cache = self.begin(state, extra, prior_discretized=prior_discretized)
        return self.complete(state, cache, output_scale=output_scale)

    def interpolate_at_t1(self, rv, extra, /, *, prior):
        del prior
        return _InterpRes((rv, extra), (rv, extra), (rv, extra))


@containers.dataclass
class _ExtraImplFilter(_ExtraImpl):
    def init(self, sol, /):
        return sol, None

    def initial_condition(self, *, prior):
        return self.ssm.normal.from_tcoeffs(prior.tcoeffs)

    def begin(self, rv, _extra, /, prior_discretized):
        cond, (p, p_inv) = prior_discretized

        rv_p = self.ssm.normal.preconditioner_apply(rv, p_inv)

        m_ext_p = self.ssm.stats.mean(rv_p)
        extrapolated_p = self.ssm.conditional.apply(m_ext_p, cond)

        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def extract(self, hidden_state, _extra, /):
        return hidden_state

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = self.ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p = self.ssm.conditional.marginalise(rv_p, (A, noise))
        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        return extrapolated, None

    def interpolate(self, state_t0, marginal_t1, dt0, dt1, output_scale, *, prior):
        # todo: by ditching marginal_t1 and dt1, this function _extrapolates
        #  (no *inter*polation happening)
        del dt1

        hidden, extra = state_t0
        prior0 = prior.discretize(dt0)
        hidden, extra = self.begin(hidden, extra, prior_discretized=prior0)
        hidden, extra = self.complete(hidden, extra, output_scale=output_scale)

        # Consistent state-types in interpolation result.
        interp = (hidden, extra)
        step_from = (marginal_t1, None)
        return _InterpRes(step_from=step_from, interpolated=interp, interp_from=interp)

    def interpolate_at_t1(self, rv, extra, /, *, prior):
        del prior
        return _InterpRes((rv, extra), (rv, extra), (rv, extra))


@containers.dataclass
class _ExtraImplFixedPoint(_ExtraImpl):
    def initial_condition(self, prior):
        rv = self.ssm.normal.from_tcoeffs(prior.tcoeffs)
        cond = self.ssm.conditional.identity(len(prior.tcoeffs))
        return stats.MarkovSeq(init=rv, conditional=cond)

    def init(self, sol: stats.MarkovSeq, /):
        return sol.init, sol.conditional

    def begin(self, rv, extra, /, prior_discretized):
        cond, (p, p_inv) = prior_discretized

        rv_p = self.ssm.normal.preconditioner_apply(rv, p_inv)

        m_ext_p = self.ssm.stats.mean(rv_p)
        extrapolated_p = self.ssm.conditional.apply(m_ext_p, cond)

        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p, extra)
        return extrapolated, cache

    def extract(self, hidden_state, extra, /):
        return stats.MarkovSeq(init=hidden_state, conditional=extra)

    def complete(self, _rv, extra, /, output_scale):
        cond, (p, p_inv), rv_p, bw0 = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = self.ssm.stats.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = self.ssm.conditional.revert(rv_p, (A, noise))
        extrapolated = self.ssm.normal.preconditioner_apply(extrapolated_p, p)
        cond = self.ssm.conditional.preconditioner_apply(cond_p, p, p_inv)

        # Merge conditionals
        cond = self.ssm.conditional.merge(bw0, cond)

        # Gather and return
        return extrapolated, cond

    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale, prior):
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
        prior0 = prior.discretize(dt0)
        extrapolated_t = self._extrapolate(
            *state_t0, output_scale=output_scale, prior_discretized=prior0
        )
        conditional_id = self.ssm.conditional.identity(prior.num_derivatives + 1)
        previous_new = (extrapolated_t[0], conditional_id)

        prior1 = prior.discretize(dt1)
        extrapolated_t1 = self._extrapolate(
            *previous_new, output_scale=output_scale, prior_discretized=prior1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        conditional_t1_to_t = extrapolated_t1[1]
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        return _InterpRes(
            step_from=(marginal_t1, conditional_t1_to_t),
            interpolated=(rv_at_t, extrapolated_t[1]),
            interp_from=previous_new,
        )

    def _extrapolate(self, state, extra, /, *, output_scale, prior_discretized):
        x, cache = self.begin(state, extra, prior_discretized=prior_discretized)
        return self.complete(x, cache, output_scale=output_scale)

    def interpolate_at_t1(self, rv, extra, /, *, prior):
        cond_identity = self.ssm.conditional.identity(prior.num_derivatives + 1)
        return _InterpRes((rv, cond_identity), (rv, extra), (rv, cond_identity))


@containers.dataclass
class _Correction:
    """Correction model interface."""

    name: str
    ode_order: int
    ssm: Any
    linearize: Callable

    def init(self, x, /):
        """Initialise the state from the solution."""
        raise NotImplementedError

    def estimate_error(self, x, /, vector_field, t):
        """Perform all elements of the correction until the error estimate."""
        raise NotImplementedError

    def complete(self, x, cache, /):
        """Complete what has been left out by `estimate_error`."""
        raise NotImplementedError

    def extract(self, x, /):
        """Extract the solution from the state."""
        raise NotImplementedError


@containers.dataclass
class _CorrectionTS(_Correction):
    def init(self, x, /):
        y = self.ssm.prototypes.observed()
        return x, y

    def estimate_error(self, x, /, vector_field, t):
        def f_wrapped(s):
            return vector_field(*s, t=t)

        A, b = self.linearize(f_wrapped, x.mean)
        observed = self.ssm.transform.marginalise(x, (A, b))

        error_estimate = _estimate_error(observed, ssm=self.ssm)
        return error_estimate, observed, {"linearization": (A, b)}

    def complete(self, x, cache, /):
        A, b = cache["linearization"]
        observed, (_gain, corrected) = self.ssm.transform.revert(x, (A, b))
        return corrected, observed

    def extract(self, x, /):
        return x


@containers.dataclass
class _CorrectionSLR(_Correction):
    def init(self, x, /):
        y = self.ssm.prototypes.observed()
        return x, y

    def estimate_error(self, x, /, vector_field, t):
        f_wrapped = functools.partial(vector_field, t=t)
        A, b = self.linearize(f_wrapped, x)
        observed = self.ssm.conditional.marginalise(x, (A, b))

        error_estimate = _estimate_error(observed, ssm=self.ssm)
        return error_estimate, observed, (A, b, f_wrapped)

    def complete(self, x, cache, /):
        # Re-linearise (because the linearisation point changed)
        *_, f_wrapped = cache
        A, b = self.linearize(f_wrapped, x)

        # Condition
        observed, (_gain, corrected) = self.ssm.conditional.revert(x, (A, b))
        return corrected, observed

    def extract(self, x, /):
        return x


def correction_ts0(*, ssm, ode_order=1) -> _Correction:
    """Zeroth-order Taylor linearisation."""
    linearize = ssm.linearise.ode_taylor_0th(ode_order=ode_order)
    return _CorrectionTS(name="TS0", ode_order=ode_order, ssm=ssm, linearize=linearize)


def correction_ts1(*, ssm, ode_order=1) -> _Correction:
    """First-order Taylor linearisation."""
    linearize = ssm.linearise.ode_taylor_1st(ode_order=ode_order)
    return _CorrectionTS(name="TS1", ode_order=ode_order, ssm=ssm, linearize=linearize)


def correction_slr0(*, ssm, cubature_fun=cubature_third_order_spherical) -> _Correction:
    """Zeroth-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_1st(cubature_fun)
    return _CorrectionSLR(ssm=ssm, ode_order=1, linearize=linearize, name="SLR0")


def correction_slr1(*, ssm, cubature_fun=cubature_third_order_spherical) -> _Correction:
    """First-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_0th(cubature_fun)
    return _CorrectionSLR(ssm=ssm, ode_order=1, linearize=linearize, name="SLR1")


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

    #
    # def extract(self, state: _StrategyState, /, *, extrapolation, correction):
    #     """Extract the solution from a state."""
    #     hidden = correction.extract(state.hidden)
    #     sol = extrapolation.extract(hidden, state.aux_extra)
    #     return state.t, sol

    def case_interpolate_at_t1(
        self, state_t1: _StrategyState, *, extrapolation, prior
    ) -> _InterpRes:
        """Process the solution in case t=t_n."""
        tmp = extrapolation.interpolate_at_t1(
            state_t1.hidden, state_t1.aux_extra, prior=prior
        )
        step_from, solution, interp_from = (
            tmp.step_from,
            tmp.interpolated,
            tmp.interp_from,
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
        self,
        t,
        *,
        s0: _StrategyState,
        s1: _StrategyState,
        output_scale,
        extrapolation,
        prior,
    ) -> _InterpRes:
        """Process the solution in case t>t_n."""
        # Interpolate
        interp = extrapolation.interpolate(
            state_t0=(s0.hidden, s0.aux_extra),
            marginal_t1=s1.hidden,
            dt0=t - s0.t,
            dt1=s1.t - t,
            output_scale=output_scale,
            prior=prior,
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


def strategy_smoother(prior, correction: _Correction, /, ssm) -> _Strategy:
    """Construct a smoother."""
    extrapolation = _ExtraImplSmoother(
        name="Smoother",
        ssm=ssm,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )
    strategy = _Strategy()
    return strategy, correction, extrapolation, prior


def strategy_fixedpoint(prior, correction: _Correction, /, ssm) -> _Strategy:
    """Construct a fixedpoint-smoother."""
    extrapolation = _ExtraImplFixedPoint(
        name="Fixed-point smoother",
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
    )
    strategy = _Strategy()
    return strategy, correction, extrapolation, prior


def strategy_filter(prior, correction: _Correction, /, *, ssm) -> _Strategy:
    """Construct a filter."""
    extrapolation = _ExtraImplFilter(
        name="Filter",
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )
    strategy = _Strategy()
    return strategy, correction, extrapolation, prior


@containers.dataclass
class _Calibration:
    """Calibration implementation."""

    init: Callable
    update: Callable
    extract: Callable


class _SolverState(containers.NamedTuple):
    """Solver state."""

    strategy: Any
    output_scale: Any

    @property
    def t(self):
        return self.strategy.t


@containers.dataclass
class _ProbabilisticSolver:
    name: str
    requires_rescaling: bool

    step_implementation: Callable

    prior: _MarkovProcess
    ssm: Any
    extrapolation: _ExtraImpl
    calibration: _Calibration
    correction: _Correction
    strategy: _Strategy

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t

        rv, extra = self.extrapolation.init(posterior_t0)
        rv, corr = self.correction.init(rv)
        state_t0 = _StrategyState(t=t0, hidden=rv, aux_extra=extra, aux_corr=corr)

        # TODO: Replace dt0, dt1, and prior with prior_dt0, and prior_dt1
        interp = self.extrapolation.interpolate(
            state_t0=(state_t0.hidden, state_t0.aux_extra),
            marginal_t1=marginals_t1,
            dt0=dt0,
            dt1=dt1,
            output_scale=output_scale,
            prior=self.prior,
        )

        (marginals, _aux) = interp.interpolated
        u = self.ssm.stats.qoi(marginals)
        return u, marginals

    @property
    def error_contraction_rate(self):
        return self.prior.num_derivatives + 1

    @property
    def is_suitable_for_offgrid_marginals(self):
        return self.extrapolation.is_suitable_for_offgrid_marginals

    @property
    def is_suitable_for_save_at(self):
        return self.extrapolation.is_suitable_for_save_at

    @property
    def is_suitable_for_save_every_step(self):
        return self.extrapolation.is_suitable_for_save_every_step

    def init(self, t, initial_condition) -> _SolverState:
        posterior, output_scale = initial_condition

        rv, extra = self.extrapolation.init(posterior)
        rv, corr = self.correction.init(rv)
        state_strategy = _StrategyState(t=t, hidden=rv, aux_extra=extra, aux_corr=corr)

        calib_state = self.calibration.init(output_scale)
        return _SolverState(strategy=state_strategy, output_scale=calib_state)

    def step(self, state: _SolverState, *, vector_field, dt) -> _SolverState:
        return self.step_implementation(
            state, vector_field=vector_field, dt=dt, calibration=self.calibration
        )

    def extract(self, state: _SolverState, /):
        hidden = self.correction.extract(state.strategy.hidden)
        posterior = self.extrapolation.extract(hidden, state.strategy.aux_extra)
        t = state.strategy.t

        _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
        return t, (posterior, output_scale)

    def interpolate(
        self, t, *, interp_from: _SolverState, interp_to: _SolverState
    ) -> _InterpRes:
        output_scale, _ = self.calibration.extract(interp_to.output_scale)
        interp = self.strategy.case_interpolate(
            t,
            s0=interp_from.strategy,
            s1=interp_to.strategy,
            output_scale=output_scale,
            extrapolation=self.extrapolation,
            prior=self.prior,
        )
        prev = _SolverState(interp.interp_from, output_scale=interp_from.output_scale)
        sol = _SolverState(interp.interpolated, output_scale=interp_to.output_scale)
        acc = _SolverState(interp.step_from, output_scale=interp_to.output_scale)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)

    def interpolate_at_t1(self, *, interp_from, interp_to) -> _InterpRes:
        x = self.strategy.case_interpolate_at_t1(
            interp_to.strategy, extrapolation=self.extrapolation, prior=self.prior
        )

        prev = _SolverState(x.interp_from, output_scale=interp_from.output_scale)
        sol = _SolverState(x.interpolated, output_scale=interp_to.output_scale)
        acc = _SolverState(x.step_from, output_scale=interp_to.output_scale)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)

    def initial_condition(self):
        """Construct an initial condition."""
        posterior = self.extrapolation.initial_condition(prior=self.prior)
        return posterior, self.prior.output_scale


def solver_mle(inputs, *, ssm):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """
    strategy, correction, extrapolation, prior = inputs

    def step_mle(state, /, *, dt, vector_field, calibration):
        output_scale_prior, _calibrated = calibration.extract(state.output_scale)

        prior_discretized = prior.discretize(dt)
        hidden, extra = extrapolation.begin(
            state.strategy.hidden,
            state.strategy.aux_extra,
            prior_discretized=prior_discretized,
        )
        t = state.t + dt
        error, _, corr = correction.estimate_error(
            hidden, vector_field=vector_field, t=t
        )

        hidden, extra = extrapolation.complete(
            hidden, extra, output_scale=output_scale_prior
        )
        hidden, corr = correction.complete(hidden, corr)
        state_strategy = _StrategyState(
            t=t, hidden=hidden, aux_extra=extra, aux_corr=corr
        )
        observed = state_strategy.aux_corr

        # Calibrate
        output_scale = calibration.update(state.output_scale, observed=observed)

        # Return
        state = _SolverState(strategy=state_strategy, output_scale=output_scale)
        return dt * error, state

    return _ProbabilisticSolver(
        ssm=ssm,
        name="Probabilistic solver with MLE calibration",
        prior=prior,
        calibration=_calibration_running_mean(ssm=ssm),
        step_implementation=step_mle,
        extrapolation=extrapolation,
        correction=correction,
        strategy=strategy,
        requires_rescaling=True,
    )


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


def solver_dynamic(strategy, *, ssm):
    """Create a solver that calibrates the output scale dynamically."""

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

    return _ProbabilisticSolver(
        strategy=strategy,
        calibration=_calibration_most_recent(ssm=ssm),
        name="Dynamic probabilistic solver",
        step_implementation=step_dynamic,
        requires_rescaling=False,
    )


def _calibration_most_recent(*, ssm) -> _Calibration:
    def init(prior):
        return prior

    def update(_state, /, observed):
        return ssm.stats.mahalanobis_norm_relative(0.0, observed)

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def solver(inputs, /, *, ssm):
    """Create a solver that does not calibrate the output scale automatically."""
    strategy, correction, extrapolation, prior = inputs

    def step(state: _SolverState, *, vector_field, dt, calibration):
        del calibration  # unused

        prior_discretized = prior.discretize(dt)
        hidden, extra = extrapolation.begin(
            state.strategy.hidden,
            state.strategy.aux_extra,
            prior_discretized=prior_discretized,
        )
        t = state.t + dt
        error, _, corr = correction.estimate_error(
            hidden, vector_field=vector_field, t=t
        )

        hidden, extra = extrapolation.complete(
            hidden, extra, output_scale=state.output_scale
        )
        hidden, corr = correction.complete(hidden, corr)
        state_strategy = _StrategyState(
            t=t, hidden=hidden, aux_extra=extra, aux_corr=corr
        )

        # Extract and return solution
        state = _SolverState(strategy=state_strategy, output_scale=state.output_scale)
        return dt * error, state

    return _ProbabilisticSolver(
        strategy=strategy,
        ssm=ssm,
        prior=prior,
        extrapolation=extrapolation,
        correction=correction,
        calibration=_calibration_none(),
        step_implementation=step,
        name="Probabilistic solver",
        requires_rescaling=False,
    )


def _calibration_none() -> _Calibration:
    def init(prior):
        return prior

    def update(_state, /, observed):
        raise NotImplementedError

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)
