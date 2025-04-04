"""Probabilistic IVP solvers."""

from probdiffeq import stats
from probdiffeq.backend import (
    containers,
    control_flow,
    functools,
    linalg,
    special,
    tree_util,
)
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Array, Callable, Generic, NamedArg, TypeVar
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

    if output_scale is None:
        output_scale = np.ones_like(ssm.prototypes.output_scale())
    discretize = ssm.conditional.ibm_transitions(output_scale=output_scale)

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


def strategy_smoother(*, ssm):
    """Construct a smoother."""
    return _ExtraImplSmoother(
        name="Smoother",
        ssm=ssm,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


def strategy_fixedpoint(*, ssm):
    """Construct a fixedpoint-smoother."""
    return _ExtraImplFixedPoint(
        name="Fixed-point smoother",
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
    )


def strategy_filter(*, ssm):
    """Construct a filter."""
    return _ExtraImplFilter(
        name="Filter",
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


@containers.dataclass
class _Calibration:
    """Calibration implementation."""

    init: Callable
    update: Callable
    extract: Callable


class _State(containers.NamedTuple):
    """Solver state."""

    t: Any
    hidden: Any
    aux_extra: Any
    output_scale: Any


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

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t

        rv, extra = self.extrapolation.init(posterior_t0)
        rv, corr = self.correction.init(rv)

        # TODO: Replace dt0, dt1, and prior with prior_dt0, and prior_dt1
        interp = self.extrapolation.interpolate(
            state_t0=(rv, extra),
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

    def init(self, t, initial_condition) -> _State:
        posterior, output_scale = initial_condition

        rv, extra = self.extrapolation.init(posterior)
        rv, corr = self.correction.init(rv)

        calib_state = self.calibration.init(output_scale)
        return _State(t=t, hidden=rv, aux_extra=extra, output_scale=calib_state)

    def step(self, state: _State, *, vector_field, dt):
        return self.step_implementation(
            state, vector_field=vector_field, dt=dt, calibration=self.calibration
        )

    def extract(self, state: _State, /):
        hidden = self.correction.extract(state.hidden)
        posterior = self.extrapolation.extract(hidden, state.aux_extra)
        t = state.t

        _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
        return t, (posterior, output_scale)

    def interpolate(self, t, *, interp_from: _State, interp_to: _State) -> _InterpRes:
        output_scale, _ = self.calibration.extract(interp_to.output_scale)
        return self._case_interpolate(
            t, s0=interp_from, s1=interp_to, output_scale=output_scale
        )

    def _case_interpolate(self, t, *, s0, s1, output_scale) -> _InterpRes:
        """Process the solution in case t>t_n."""
        # Interpolate
        interp = self.extrapolation.interpolate(
            state_t0=(s0.hidden, s0.aux_extra),
            marginal_t1=s1.hidden,
            dt0=t - s0.t,
            dt1=s1.t - t,
            output_scale=output_scale,
            prior=self.prior,
        )

        # Turn outputs into valid states

        def _state(t_, x, scale):
            return _State(t=t_, hidden=x[0], aux_extra=x[1], output_scale=scale)

        step_from = _state(s1.t, interp.step_from, s1.output_scale)
        interpolated = _state(t, interp.interpolated, s1.output_scale)
        interp_from = _state(t, interp.interp_from, s0.output_scale)
        return _InterpRes(
            step_from=step_from, interpolated=interpolated, interp_from=interp_from
        )

    def interpolate_at_t1(self, *, interp_from, interp_to) -> _InterpRes:
        """Process the solution in case t=t_n."""
        tmp = self.extrapolation.interpolate_at_t1(
            interp_to.hidden, interp_to.aux_extra, prior=self.prior
        )
        step_from_, solution_, interp_from_ = (
            tmp.step_from,
            tmp.interpolated,
            tmp.interp_from,
        )

        def _state(t_, s, scale):
            return _State(t=t_, hidden=s[0], aux_extra=s[1], output_scale=scale)

        t = interp_to.t
        prev = _state(t, interp_from_, interp_from.output_scale)
        sol = _state(t, solution_, interp_to.output_scale)
        acc = _state(t, step_from_, interp_to.output_scale)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)

    def initial_condition(self):
        """Construct an initial condition."""
        posterior = self.extrapolation.initial_condition(prior=self.prior)
        return posterior, self.prior.output_scale


def solver_mle(extrapolation, /, *, correction, prior, ssm):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """

    def step_mle(state, /, *, dt, vector_field, calibration):
        output_scale_prior, _calibrated = calibration.extract(state.output_scale)

        prior_discretized = prior.discretize(dt)
        hidden, extra = extrapolation.begin(
            state.hidden, state.aux_extra, prior_discretized=prior_discretized
        )
        t = state.t + dt
        error, _, corr = correction.estimate_error(
            hidden, vector_field=vector_field, t=t
        )

        hidden, extra = extrapolation.complete(
            hidden, extra, output_scale=output_scale_prior
        )
        hidden, observed = correction.complete(hidden, corr)

        output_scale = calibration.update(state.output_scale, observed=observed)
        state = _State(t=t, hidden=hidden, aux_extra=extra, output_scale=output_scale)
        return dt * error, state

    return _ProbabilisticSolver(
        ssm=ssm,
        name="Probabilistic solver with MLE calibration",
        prior=prior,
        calibration=_calibration_running_mean(ssm=ssm),
        step_implementation=step_mle,
        extrapolation=extrapolation,
        correction=correction,
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


def solver_dynamic(extrapolation, *, correction, prior, ssm):
    """Create a solver that calibrates the output scale dynamically."""

    def step_dynamic(state, /, *, dt, vector_field, calibration):
        prior_discretized = prior.discretize(dt)
        hidden, extra = extrapolation.begin(
            state.hidden, state.aux_extra, prior_discretized=prior_discretized
        )
        t = state.t + dt
        error, observed, corr = correction.estimate_error(
            hidden, vector_field=vector_field, t=t
        )

        output_scale = calibration.update(state.output_scale, observed=observed)

        prior_, _calibrated = calibration.extract(output_scale)
        hidden, extra = extrapolation.complete(hidden, extra, output_scale=prior_)
        hidden, corr = correction.complete(hidden, corr)

        # Return solution
        state = _State(t=t, hidden=hidden, aux_extra=extra, output_scale=output_scale)
        return dt * error, state

    return _ProbabilisticSolver(
        prior=prior,
        ssm=ssm,
        extrapolation=extrapolation,
        correction=correction,
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


def solver(extrapolation, /, *, correction, prior, ssm):
    """Create a solver that does not calibrate the output scale automatically."""

    def step(state: _State, *, vector_field, dt, calibration):
        del calibration  # unused

        prior_discretized = prior.discretize(dt)
        hidden, extra = extrapolation.begin(
            state.hidden, state.aux_extra, prior_discretized=prior_discretized
        )
        t = state.t + dt
        error, _, corr = correction.estimate_error(
            hidden, vector_field=vector_field, t=t
        )

        hidden, extra = extrapolation.complete(
            hidden, extra, output_scale=state.output_scale
        )
        hidden, corr = correction.complete(hidden, corr)

        # Extract and return solution
        state = _State(
            t=t, hidden=hidden, aux_extra=extra, output_scale=state.output_scale
        )
        return dt * error, state

    return _ProbabilisticSolver(
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


def adaptive(slvr, /, *, ssm, atol=1e-4, rtol=1e-2, control=None, norm_ord=None):
    """Make an IVP solver adaptive."""
    if control is None:
        control = control_proportional_integral()

    return _AdaSolver(
        slvr, ssm=ssm, atol=atol, rtol=rtol, control=control, norm_ord=norm_ord
    )


class _AdaState(containers.NamedTuple):
    step_from: Any
    interp_from: Any
    control: Any
    stats: Any


class _AdaSolver:
    """Adaptive IVP solvers."""

    def __init__(
        self, slvr: _ProbabilisticSolver, /, *, atol, rtol, control, norm_ord, ssm
    ):
        self.solver = slvr
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.ssm = ssm

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            "\n)"
        )

    @functools.jit
    def init(self, t, initial_condition, dt, num_steps) -> _AdaState:
        """Initialise the IVP solver state."""
        state_solver = self.solver.init(t, initial_condition)
        state_control = self.control.init(dt)
        return _AdaState(state_solver, state_solver, state_control, num_steps)

    @functools.jit
    def rejection_loop(self, state0: _AdaState, *, vector_field, t1) -> _AdaState:
        class _RejectionState(containers.NamedTuple):
            """State for rejection loops.

            (Keep decreasing step-size until error norm is small.
            This is one part of an IVP solver step.)
            """

            error_norm_proposed: float
            control: Any
            proposed: Any
            step_from: Any

        def init(s0: _AdaState) -> _RejectionState:
            def _inf_like(tree):
                return tree_util.tree_map(lambda x: np.inf() * np.ones_like(x), tree)

            smaller_than_1 = 1.0 / 1.1  # the cond() must return True
            return _RejectionState(
                error_norm_proposed=smaller_than_1,
                control=s0.control,
                proposed=_inf_like(s0.step_from),
                step_from=s0.step_from,
            )

        def cond_fn(state: _RejectionState) -> bool:
            # error_norm_proposed is EEst ** (-1/rate), thus "<"
            return state.error_norm_proposed < 1.0

        def body_fn(state: _RejectionState) -> _RejectionState:
            """Attempt a step.

            Perform a step with an IVP solver and
            propose a future time-step based on tolerances and error estimates.
            """
            # Some controllers like to clip the terminal value instead of interpolating.
            # This must happen _before_ the step.
            state_control = self.control.clip(state.control, t=state.step_from.t, t1=t1)

            # Perform the actual step.
            error_estimate, state_proposed = self.solver.step(
                state=state.step_from,
                vector_field=vector_field,
                dt=self.control.extract(state_control),
            )
            # Normalise the error
            u_proposed = self.ssm.stats.qoi(state_proposed.hidden)[0]
            u_step_from = self.ssm.stats.qoi(state_proposed.hidden)[0]
            u = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
            error_power = _error_scale_and_normalize(error_estimate, u=u)

            # Propose a new step
            state_control = self.control.apply(state_control, error_power=error_power)
            return _RejectionState(
                error_norm_proposed=error_power,  # new
                proposed=state_proposed,  # new
                control=state_control,  # new
                step_from=state.step_from,
            )

        def _error_scale_and_normalize(error_estimate, *, u):
            error_relative = error_estimate / (self.atol + self.rtol * np.abs(u))
            dim = np.atleast_1d(u).size
            error_norm = linalg.vector_norm(error_relative, order=self.norm_ord)
            error_norm_rel = error_norm / np.sqrt(dim)
            return error_norm_rel ** (-1.0 / self.solver.error_contraction_rate)

        def extract(s: _RejectionState) -> _AdaState:
            num_steps = state0.stats + 1
            return _AdaState(s.proposed, s.step_from, s.control, num_steps)

        init_val = init(state0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def extract_before_t1(self, state: _AdaState):
        solution_solver = self.solver.extract(state.step_from)
        solution_control = self.control.extract(state.control)
        return solution_solver, solution_control, state.stats

    def extract_at_t1(self, state: _AdaState):
        # todo: make the "at t1" decision inside interpolate(),
        #  which collapses the next two functions together
        interp = self.solver.interpolate_at_t1(
            interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            interp.step_from, interp.interp_from, state.control, state.stats
        )

        solution_solver = self.solver.extract(interp.interpolated)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)

    def extract_after_t1_via_interpolation(self, state: _AdaState, t):
        interp = self.solver.interpolate(
            t, interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            interp.step_from, interp.interp_from, state.control, state.stats
        )

        solution_solver = self.solver.extract(interp.interpolated)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)

    @staticmethod
    def register_pytree_node():
        def _asolver_flatten(asolver):
            children = (asolver.atol, asolver.rtol)
            aux = (asolver.solver, asolver.control, asolver.norm_ord, asolver.ssm)
            return children, aux

        def _asolver_unflatten(aux, children):
            atol, rtol = children
            (slvr, control, norm_ord, ssm) = aux
            return _AdaSolver(
                slvr, atol=atol, rtol=rtol, control=control, norm_ord=norm_ord, ssm=ssm
            )

        tree_util.register_pytree_node(
            _AdaSolver, flatten_func=_asolver_flatten, unflatten_func=_asolver_unflatten
        )


_AdaSolver.register_pytree_node()


@containers.dataclass
class _Controller:
    """Control algorithm."""

    init: Callable[[float], Any]
    """Initialise the controller state."""

    clip: Callable[[Any, float, float], Any]
    """(Optionally) clip the current step to not exceed t1."""

    apply: Callable[[Any, NamedArg(float, "error_power")], Any]
    r"""Propose a time-step $\Delta t$."""

    extract: Callable[[Any], float]
    """Extract the time-step from the controller state."""


def control_proportional_integral(
    *,
    clip: bool = False,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _Controller:
    """Construct a proportional-integral-controller with time-clipping."""

    class PIState(containers.NamedTuple):
        dt: float
        error_power_previously_accepted: float

    def init(dt: float, /) -> PIState:
        return PIState(dt, 1.0)

    def apply(state: PIState, /, *, error_power) -> PIState:
        # error_power = error_norm ** (-1.0 / error_contraction_rate)
        dt_proposed, error_power_prev = state

        a1 = error_power**power_integral_unscaled
        a2 = (error_power / error_power_prev) ** power_proportional_unscaled
        scale_factor_unclipped = safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt_proposed
        return PIState(dt_proposed, error_power_prev)

    def extract(state: PIState, /) -> float:
        dt_proposed, _error_norm_previously_accepted = state
        return dt_proposed

    if clip:

        def clip_fun(state: PIState, /, t, t1) -> PIState:
            dt_proposed, error_norm_previously_accepted = state
            dt = dt_proposed
            dt_clipped = np.minimum(dt, t1 - t)
            return PIState(dt_clipped, error_norm_previously_accepted)

        return _Controller(init=init, apply=apply, extract=extract, clip=clip_fun)

    return _Controller(init=init, apply=apply, extract=extract, clip=lambda v, **_kw: v)


def control_integral(
    *, clip=False, safety=0.95, factor_min=0.2, factor_max=10.0
) -> _Controller:
    """Construct an integral-controller."""

    def init(dt, /):
        return dt

    def apply(dt, /, *, error_power):
        # error_power = error_norm ** (-1.0 / error_contraction_rate)
        scale_factor_unclipped = safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
        return scale_factor * dt

    def extract(dt, /):
        return dt

    if clip:

        def clip_fun(dt, /, t, t1):
            return np.minimum(dt, t1 - t)

        return _Controller(init=init, apply=apply, extract=extract, clip=clip_fun)

    return _Controller(init=init, apply=apply, extract=extract, clip=lambda v, **_kw: v)
