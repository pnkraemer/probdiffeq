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
from probdiffeq.backend.typing import (
    Any,
    ArrayLike,
    Callable,
    Generic,
    NamedArg,
    TypeVar,
)
from probdiffeq.impl import impl


def prior_wiener_integrated(
    tcoeffs, *, ssm_fact: str, output_scale: ArrayLike | None = None, damp: float = 0.0
):
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process."""
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs)

    # TODO: should the output_scale be an argument to solve()?
    # TODO: should the output scale (and all 'damp'-like factors)
    #       mirror the pytree structure of 'tcoeffs'?
    if output_scale is None:
        output_scale = np.ones_like(ssm.prototypes.output_scale())

    discretize = ssm.conditional.ibm_transitions(base_scale=output_scale)

    # Increase damping to get visually more pleasing uncertainties
    #  and more numerical robustness for
    #  high-order solvers in low precision arithmetic
    init = ssm.normal.from_tcoeffs(tcoeffs, damp=damp)
    return init, discretize, ssm


def prior_wiener_integrated_discrete(ts, *args, **kwargs):
    """Compute a time-discretized, multiply-integrated Wiener process."""
    init, discretize, ssm = prior_wiener_integrated(*args, **kwargs)
    scales = np.ones_like(ssm.prototypes.output_scale())
    discretize_vmap = functools.vmap(discretize, in_axes=(0, None))
    conditionals = discretize_vmap(np.diff(ts), scales)
    return init, conditionals, ssm


R = TypeVar("R")


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

    points: ArrayLike
    weights_sqrtm: ArrayLike


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


# TODO: how does this generalise to an input_shape instead of an input_dimension?
#  via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return np.prod_along_axis(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = np.meshgrid(*([x] * d))
    y_mesh = tree_util.tree_map(lambda s: np.reshape(s, (-1,)), x_mesh)
    return np.stack(y_mesh).T


@containers.dataclass
class _Strategy:
    """Estimation-strategy interface."""

    ssm: Any

    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int
    is_suitable_for_offgrid_marginals: int

    def init(self, sol, /):
        """Initialise a state from a solution."""
        raise NotImplementedError

    def extrapolate(self, rv, strategy_state, /, *, transition):
        """Extrapolate (also known as prediction)."""
        raise NotImplementedError

    def extract(self, rv, strategy_state, /):
        """Extract a solution from a state."""
        raise NotImplementedError

    def interpolate(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
        """Interpolate."""
        raise NotImplementedError

    def interpolate_at_t1(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
        """Process the state at a checkpoint."""
        raise NotImplementedError


def strategy_smoother(*, ssm) -> _Strategy:
    """Construct a smoother."""

    @containers.dataclass
    class Smoother(_Strategy):
        def init(self, sol, /):
            # Special case for implementing offgrid-marginals...
            if isinstance(sol, stats.MarkovSeq):
                rv = sol.init
                cond = sol.conditional
            else:
                rv = sol
                cond = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            return rv, cond

        def extrapolate(self, rv, aux, /, *, transition):
            del aux
            return self.ssm.conditional.revert(rv, transition)

        def extract(self, hidden_state, extra, /):
            return stats.MarkovSeq(init=hidden_state, conditional=extra)

        def interpolate(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
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
            # TODO: if we pass prior1 and prior2, then
            #       we don't have to pass dt0, dt1, output_scale, and prior...

            # Extrapolate from t0 to t, and from t to t1.
            prior0 = prior(dt0, output_scale)
            extrapolated_t = self.extrapolate(*state_t0, transition=prior0)
            prior1 = prior(dt1, output_scale)
            extrapolated_t1 = self.extrapolate(*extrapolated_t, transition=prior1)

            # Marginalise from t1 to t to obtain the interpolated solution.
            marginal_t1, _ = state_t1
            conditional_t1_to_t = extrapolated_t1[1]
            rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
            solution_at_t = (rv_at_t, extrapolated_t[1])

            # The state at t1 gets a new backward model;
            # (it must remember how to get back to t, not to t0).
            solution_at_t1 = (marginal_t1, conditional_t1_to_t)
            return _InterpRes(
                step_from=solution_at_t1,
                interpolated=solution_at_t,
                interp_from=solution_at_t,
            )

        def interpolate_at_t1(
            self, state_t0, state_t1, *, dt0, dt1, output_scale, prior
        ):
            del prior
            del state_t0
            del dt0
            del dt1
            del output_scale
            return _InterpRes(state_t1, state_t1, state_t1)

    return Smoother(
        ssm=ssm,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


def strategy_filter(*, ssm) -> _Strategy:
    """Construct a filter."""

    @containers.dataclass
    class Filter(_Strategy):
        def init(self, sol, /):
            return sol, None

        def extrapolate(self, rv, aux, /, *, transition):
            del aux
            rv = self.ssm.conditional.marginalise(rv, transition)

            return rv, None

        def extract(self, hidden_state, _extra, /):
            return hidden_state

        def interpolate(self, state_t0, state_t1, dt0, dt1, output_scale, *, prior):
            # todo: by ditching marginal_t1 and dt1, this function _extrapolates
            #  (no *inter*polation happening)
            del dt1
            marginal_t1, _ = state_t1

            hidden, extra = state_t0
            prior0 = prior(dt0, output_scale)
            hidden, extra = self.extrapolate(hidden, extra, transition=prior0)

            # Consistent state-types in interpolation result.
            interp = (hidden, extra)
            step_from = (marginal_t1, None)
            return _InterpRes(
                step_from=step_from, interpolated=interp, interp_from=interp
            )

        def interpolate_at_t1(
            self, state_t0, state_t1, dt0, dt1, output_scale, *, prior
        ):
            del prior
            del state_t0
            del dt0
            del dt1
            del output_scale
            rv, extra = state_t1
            return _InterpRes((rv, extra), (rv, extra), (rv, extra))

    return Filter(
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


def strategy_fixedpoint(*, ssm) -> _Strategy:
    """Construct a fixedpoint-smoother."""

    @containers.dataclass
    class FixedPoint(_Strategy):
        def init(self, sol, /):
            cond = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            return sol, cond

        def extrapolate(self, rv, bw0, /, *, transition):
            extrapolated, cond = self.ssm.conditional.revert(rv, transition)
            cond = self.ssm.conditional.merge(bw0, cond)
            return extrapolated, cond

        def extract(self, hidden_state, extra, /):
            return stats.MarkovSeq(init=hidden_state, conditional=extra)

        def interpolate_at_t1(
            self, state_t0, state_t1, *, dt0, dt1, output_scale, prior
        ):
            del prior
            del state_t0
            del dt0
            del dt1
            del output_scale
            rv, extra = state_t1
            cond_identity = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            return _InterpRes((rv, cond_identity), (rv, extra), (rv, cond_identity))

        def interpolate(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
            """Interpolate.

            A fixed-point smoother interpolates by

            * Extrapolating from t0 to t, which gives the "filtering" marginal
            and the backward transition from t to t0.
            * Extrapolating from t to t1, which gives another "filtering" marginal
            and the backward transition from t1 to t.
            * Applying the t1-to-t backward transition
            to compute the interpolation result.
            This intermediate result is informed about its "right-hand side" datum.

            The difference to smoother-interpolation is quite subtle:

            * The backward transition of the solution at 't'
            is merged with that at 't0'.
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
            marginal_t1, _ = state_t1
            # Extrapolate from t0 to t, and from t to t1.
            # This yields all building blocks.
            prior0 = prior(dt0, output_scale)
            extrapolated_t = self.extrapolate(*state_t0, transition=prior0)
            conditional_id = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            previous_new = (extrapolated_t[0], conditional_id)

            prior1 = prior(dt1, output_scale)
            extrapolated_t1 = self.extrapolate(*previous_new, transition=prior1)

            # Marginalise from t1 to t to obtain the interpolated solution.
            conditional_t1_to_t = extrapolated_t1[1]
            rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

            # Return the right combination of marginals and conditionals.
            return _InterpRes(
                step_from=(marginal_t1, conditional_t1_to_t),
                interpolated=(rv_at_t, extrapolated_t[1]),
                interp_from=previous_new,
            )

    return FixedPoint(
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
    )


@containers.dataclass
class _Correction:
    """Correction model interface."""

    name: str
    ode_order: int
    ssm: Any
    linearize: Any
    vector_field: Callable
    re_linearize: bool

    def init(self, x, /):
        """Initialise the state from the solution."""
        jac = self.linearize.init()
        return x, jac

    def estimate_error(self, rv, correction_state, /, t):
        """Estimate the error."""
        f_wrapped = functools.partial(self.vector_field, t=t)
        cond, correction_state = self.linearize.update(f_wrapped, rv, correction_state)
        observed = self.ssm.conditional.marginalise(rv, cond)

        zero_data = np.zeros(())
        output_scale = self.ssm.stats.mahalanobis_norm_relative(zero_data, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)
        error_estimate_unscaled = np.squeeze(stdev)
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, observed, (correction_state, cond)

    def correct(self, rv, correction_state, /, t):
        """Perform the correction step."""
        linearization_state, cond = correction_state

        if self.re_linearize:
            f_wrapped = functools.partial(self.vector_field, t=t)
            cond, linearization_state = self.linearize.update(
                f_wrapped, rv, linearization_state
            )

        observed, reverted = self.ssm.conditional.revert(rv, cond)
        corrected = reverted.noise
        return corrected, observed, linearization_state


def correction_ts0(vector_field, *, ssm, ode_order=1, damp: float = 0.0) -> _Correction:
    """Zeroth-order Taylor linearisation."""
    linearize = ssm.linearise.ode_taylor_0th(ode_order=ode_order, damp=damp)
    return _Correction(
        name="TS0",
        vector_field=vector_field,
        ode_order=ode_order,
        ssm=ssm,
        linearize=linearize,
        re_linearize=False,
    )


def correction_ts1(
    vector_field,
    *,
    ssm,
    ode_order=1,
    damp: float = 0.0,
    jvp_probes=10,
    jvp_probes_seed=1,
) -> _Correction:
    """First-order Taylor linearisation."""
    assert jvp_probes > 0
    linearize = ssm.linearise.ode_taylor_1st(
        ode_order=ode_order,
        damp=damp,
        jvp_probes=jvp_probes,
        jvp_probes_seed=jvp_probes_seed,
    )
    return _Correction(
        name="TS1",
        vector_field=vector_field,
        ode_order=ode_order,
        ssm=ssm,
        linearize=linearize,
        re_linearize=False,
    )


def correction_slr0(
    vector_field, *, ssm, cubature_fun=cubature_third_order_spherical, damp: float = 0.0
) -> _Correction:
    """Zeroth-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_0th(cubature_fun, damp=damp)
    return _Correction(
        ssm=ssm,
        vector_field=vector_field,
        ode_order=1,
        linearize=linearize,
        name="SLR0",
        re_linearize=True,
    )


def correction_slr1(
    vector_field, *, ssm, cubature_fun=cubature_third_order_spherical, damp: float = 0.0
) -> _Correction:
    """First-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_1st(cubature_fun, damp=damp)
    return _Correction(
        ssm=ssm,
        vector_field=vector_field,
        ode_order=1,
        linearize=linearize,
        name="SLR1",
        re_linearize=True,
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
    rv: Any
    strategy_state: Any
    correction_state: Any
    output_scale: Any


@tree_util.register_dataclass
@containers.dataclass
class _ErrorEstimate:
    estimate: ArrayLike
    reference: ArrayLike


@containers.dataclass
class _ProbabilisticSolver:
    name: str

    step_implementation: Callable

    prior: Callable
    ssm: Any
    strategy: _Strategy
    calibration: _Calibration
    correction: _Correction

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t
        rv, extra = self.strategy.init(posterior_t0)
        rv, corr = self.correction.init(rv)

        # TODO: Replace dt0, dt1, prior, and output_scale with prior_dt0, and prior_dt1
        interp = self.strategy.interpolate(
            state_t0=(rv, extra),
            state_t1=(marginals_t1, None),
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
        return self.ssm.num_derivatives + 1

    @property
    def is_suitable_for_offgrid_marginals(self):
        return self.strategy.is_suitable_for_offgrid_marginals

    @property
    def is_suitable_for_save_at(self):
        return self.strategy.is_suitable_for_save_at

    @property
    def is_suitable_for_save_every_step(self):
        return self.strategy.is_suitable_for_save_every_step

    def init(self, t, init) -> _State:
        rv, extra = self.strategy.init(init)
        rv, corr = self.correction.init(rv)

        # TODO: make the init() and extract() an interface.
        #       Then, lots of calibration logic simplifies considerably.
        calib_state = self.calibration.init()
        return _State(
            t=t,
            rv=rv,
            strategy_state=extra,
            correction_state=corr,
            output_scale=calib_state,
        )

    def step(self, state: _State, *, dt):
        return self.step_implementation(state, dt=dt, calibration=self.calibration)

    def extract(self, state: _State, /):
        posterior = self.strategy.extract(state.rv, state.strategy_state)
        t = state.t

        _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
        return t, (posterior, output_scale)

    def interpolate(self, *, t, interp_from: _State, interp_to: _State) -> _InterpRes:
        output_scale, _ = self.calibration.extract(interp_to.output_scale)

        # Interpolate
        interp = self.strategy.interpolate(
            state_t0=(interp_from.rv, interp_from.strategy_state),
            state_t1=(interp_to.rv, interp_to.strategy_state),
            dt0=t - interp_from.t,
            dt1=interp_to.t - t,
            output_scale=output_scale,
            prior=self.prior,
        )

        # Turn outputs into valid states

        def _state(t_, x, scale, cs):
            return _State(
                t=t_,
                rv=x[0],
                strategy_state=x[1],
                correction_state=cs,
                output_scale=scale,
            )

        step_from = _state(
            interp_to.t,
            interp.step_from,
            interp_to.output_scale,
            interp_to.correction_state,
        )
        interpolated = _state(
            t, interp.interpolated, interp_to.output_scale, interp_to.correction_state
        )
        interp_from = _state(
            t,
            interp.interp_from,
            interp_from.output_scale,
            interp_from.correction_state,
        )
        return _InterpRes(
            step_from=step_from, interpolated=interpolated, interp_from=interp_from
        )

    def interpolate_at_t1(
        self, *, t, interp_from: _State, interp_to: _State
    ) -> _InterpRes:
        """Process the solution in case t=t_n."""
        del t
        tmp = self.strategy.interpolate_at_t1(
            state_t0=None,
            dt0=None,
            dt1=None,
            output_scale=None,
            state_t1=(interp_to.rv, interp_to.strategy_state),
            prior=self.prior,
        )
        step_from_, solution_, interp_from_ = (
            tmp.step_from,
            tmp.interpolated,
            tmp.interp_from,
        )

        def _state(t_, x, scale, cs):
            return _State(
                t=t_,
                rv=x[0],
                strategy_state=x[1],
                correction_state=cs,
                output_scale=scale,
            )

        t = interp_to.t
        prev = _state(
            t, interp_from_, interp_from.output_scale, interp_from.correction_state
        )
        sol = _state(t, solution_, interp_to.output_scale, interp_to.correction_state)
        acc = _state(t, step_from_, interp_to.output_scale, interp_to.correction_state)
        return _InterpRes(step_from=acc, interpolated=sol, interp_from=prev)


def solver_mle(strategy, *, correction, prior, ssm):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """

    def step_mle(state, /, *, dt, calibration):
        u_step_from = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]

        # Estimate the error
        output_scale_prior, _calibrated = calibration.extract(state.output_scale)
        transition = prior(dt, output_scale_prior)
        mean = ssm.stats.mean(state.rv)
        mean_extra = ssm.conditional.apply(mean, transition)
        t = state.t + dt
        error, _, correction_state = correction.estimate_error(
            mean_extra, state.correction_state, t=t
        )

        # Do the full prediction step (reuse previous discretisation)
        hidden, extra = strategy.extrapolate(
            state.rv, state.strategy_state, transition=transition
        )

        # Do the full correction step
        hidden, observed, corr_state = correction.correct(hidden, correction_state, t=t)

        # Calibrate the output scale
        output_scale = calibration.update(state.output_scale, observed=observed)

        # Normalise the error

        state = _State(
            t=t,
            rv=hidden,
            strategy_state=extra,
            correction_state=corr_state,
            output_scale=output_scale,
        )
        u_proposed = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]
        reference = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    return _ProbabilisticSolver(
        ssm=ssm,
        name="Probabilistic solver with MLE calibration",
        prior=prior,
        calibration=_calibration_running_mean(ssm=ssm),
        step_implementation=step_mle,
        strategy=strategy,
        correction=correction,
    )


def _calibration_running_mean(*, ssm) -> _Calibration:
    def init():
        prior = np.ones_like(ssm.prototypes.output_scale())
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


def solver_dynamic(strategy, *, correction, prior, ssm):
    """Create a solver that calibrates the output scale dynamically."""

    def step_dynamic(state, /, *, dt, calibration):
        u_step_from = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]

        # Estimate error and calibrate the output scale
        ones = np.ones_like(ssm.prototypes.output_scale())
        transition = prior(dt, ones)
        mean = ssm.stats.mean(state.rv)
        hidden = ssm.conditional.apply(mean, transition)

        t = state.t + dt
        error, observed, correction_state = correction.estimate_error(
            hidden, state.correction_state, t=t
        )
        output_scale = calibration.update(state.output_scale, observed=observed)

        # Do the full extrapolation with the calibrated output scale
        scale, _ = calibration.extract(output_scale)
        transition = prior(dt, scale)
        hidden, extra = strategy.extrapolate(
            state.rv, state.strategy_state, transition=transition
        )

        # Do the full correction step
        hidden, _, correction_state = correction.correct(hidden, correction_state, t=t)

        # Return solution
        state = _State(
            t=t,
            rv=hidden,
            strategy_state=extra,
            correction_state=correction_state,
            output_scale=output_scale,
        )

        # Normalise the error
        u_proposed = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]
        reference = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    return _ProbabilisticSolver(
        prior=prior,
        ssm=ssm,
        strategy=strategy,
        correction=correction,
        calibration=_calibration_most_recent(ssm=ssm),
        name="Dynamic probabilistic solver",
        step_implementation=step_dynamic,
    )


def _calibration_most_recent(*, ssm) -> _Calibration:
    def init():
        return np.ones_like(ssm.prototypes.output_scale())

    def update(_state, /, observed):
        return ssm.stats.mahalanobis_norm_relative(0.0, observed)

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def solver(strategy, *, correction, prior, ssm):
    """Create a solver that does not calibrate the output scale automatically."""

    def step(state: _State, *, dt, calibration):
        del calibration  # unused

        u_step_from = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]

        # Estimate the error
        transition = prior(dt, state.output_scale)
        mean = ssm.stats.mean(state.rv)
        hidden = ssm.conditional.apply(mean, transition)
        t = state.t + dt
        error, _, correction_state = correction.estimate_error(
            hidden, state.correction_state, t=t
        )

        # Do the full extrapolation step (reuse the transition)
        hidden, extra = strategy.extrapolate(
            state.rv, state.strategy_state, transition=transition
        )

        # Do the full correction step
        hidden, _, correction_state = correction.correct(hidden, correction_state, t=t)
        state = _State(
            t=t,
            rv=hidden,
            strategy_state=extra,
            correction_state=correction_state,
            output_scale=state.output_scale,
        )

        # Normalise the error
        u_proposed = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]
        reference = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    return _ProbabilisticSolver(
        ssm=ssm,
        prior=prior,
        strategy=strategy,
        correction=correction,
        calibration=_calibration_none(ssm=ssm),
        step_implementation=step,
        name="Probabilistic solver",
    )


def _calibration_none(*, ssm) -> _Calibration:
    def init():
        return np.ones_like(ssm.prototypes.output_scale())

    def update(_state, /, observed):
        raise NotImplementedError

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def adaptive(
    slvr,
    /,
    *,
    ssm,
    atol=1e-4,
    rtol=1e-2,
    control=None,
    norm_ord=None,
    clip_dt: bool = False,
    eps: float | None = None,
):
    """Make an IVP solver adaptive."""
    if control is None:
        control = control_proportional_integral()
    if eps is None:
        eps = 10 * np.finfo_eps(float)
    return _AdaSolver(
        slvr,
        ssm=ssm,
        atol=atol,
        rtol=rtol,
        control=control,
        norm_ord=norm_ord,
        clip_dt=clip_dt,
        eps=eps,
    )


class _AdaState(containers.NamedTuple):
    dt: float
    step_from: Any
    interp_from: Any
    control: Any
    stats: Any


class _AdaSolver:
    """Adaptive IVP solvers."""

    def __init__(
        self,
        slvr: _ProbabilisticSolver,
        /,
        *,
        atol,
        rtol,
        control,
        norm_ord,
        ssm,
        clip_dt: bool,
        eps: float,
    ):
        self.solver = slvr
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.ssm = ssm
        self.clip_dt = clip_dt
        self.eps = eps

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
        return _AdaState(
            dt=dt,
            step_from=state_solver,
            interp_from=state_solver,
            control=state_control,
            stats=num_steps,
        )

    @functools.jit
    def rejection_loop(self, state0: _AdaState, *, t1) -> _AdaState:
        class _RejectionState(containers.NamedTuple):
            """State for rejection loops.

            (Keep decreasing step-size until error norm is small.
            This is one part of an IVP solver step.)
            """

            dt: float
            error_norm_proposed: float
            control: Any
            proposed: Any
            step_from: Any

        def init(s0: _AdaState) -> _RejectionState:
            def _ones_like(tree):
                return tree_util.tree_map(np.ones_like, tree)

            smaller_than_1 = 0.9  # the cond() must return True
            return _RejectionState(
                error_norm_proposed=smaller_than_1,
                dt=s0.dt,
                control=s0.control,
                proposed=_ones_like(s0.step_from),  # irrelevant
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
            dt = state.dt

            # Some controllers like to clip the terminal value instead of interpolating.
            # This must happen _before_ the step.
            if self.clip_dt:
                dt = np.minimum(dt, t1 - state.step_from.t)

            # Perform the actual step.
            error_estimate, state_proposed = self.solver.step(
                state=state.step_from, dt=dt
            )

            # Propose a new step
            error_power = self._error_scale_and_normalize(error_estimate)
            dt, state_control = self.control.apply(
                dt, state.control, error_power=error_power
            )
            return _RejectionState(
                dt=dt,
                error_norm_proposed=error_power,  # new
                proposed=state_proposed,  # new
                control=state_control,  # new
                step_from=state.step_from,
            )

        def extract(s: _RejectionState) -> _AdaState:
            num_steps = state0.stats + 1.0  # TODO: track step attempts as well
            return _AdaState(
                dt=s.dt,
                step_from=s.proposed,
                interp_from=s.step_from,
                control=s.control,
                stats=num_steps,
            )

        init_val = init(state0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def _error_scale_and_normalize(self, error: _ErrorEstimate):
        assert isinstance(error, _ErrorEstimate)
        normalize = self.atol + self.rtol * np.abs(error.reference)
        error_relative = error.estimate / normalize

        dim = np.atleast_1d(error.reference).size
        error_norm = linalg.vector_norm(error_relative, order=self.norm_ord)
        error_norm_rel = error_norm / np.sqrt(dim)
        return error_norm_rel ** (-1.0 / self.solver.error_contraction_rate)

    def extract_before_t1(self, state: _AdaState, t):
        del t
        solution_solver = self.solver.extract(state.step_from)
        extracted = solution_solver, (state.dt, state.control), state.stats
        return state, extracted

    def extract_at_t1(self, state: _AdaState, t):
        # todo: make the "at t1" decision inside solver.interpolate(),
        #  which collapses the next two functions together
        interp = self.solver.interpolate_at_t1(
            t=t, interp_from=state.interp_from, interp_to=state.step_from
        )
        return self._extract_interpolate(interp, state)

    def extract_after_t1(self, state: _AdaState, t):
        interp = self.solver.interpolate(
            t=t, interp_from=state.interp_from, interp_to=state.step_from
        )
        return self._extract_interpolate(interp, state)

    def _extract_interpolate(self, interp, state):
        state = _AdaState(
            dt=state.dt,
            step_from=interp.step_from,
            interp_from=interp.interp_from,
            control=state.control,
            stats=state.stats,
        )

        solution_solver = self.solver.extract(interp.interpolated)
        return state, (solution_solver, (state.dt, state.control), state.stats)

    @staticmethod
    def register_pytree_node():
        def _asolver_flatten(asolver):
            children = (asolver.atol, asolver.rtol, asolver.eps)
            aux = (
                asolver.solver,
                asolver.control,
                asolver.norm_ord,
                asolver.ssm,
                asolver.clip_dt,
            )
            return children, aux

        def _asolver_unflatten(aux, children):
            atol, rtol, eps = children
            (slvr, control, norm_ord, ssm, clip_dt) = aux
            return _AdaSolver(
                slvr,
                atol=atol,
                rtol=rtol,
                control=control,
                norm_ord=norm_ord,
                ssm=ssm,
                clip_dt=clip_dt,
                eps=eps,
            )

        tree_util.register_pytree_node(
            _AdaSolver, flatten_func=_asolver_flatten, unflatten_func=_asolver_unflatten
        )


_AdaSolver.register_pytree_node()

T = TypeVar("T")


@containers.dataclass
class _Controller(Generic[T]):
    """Control algorithm."""

    init: Callable[[float], T]
    """Initialise the controller state."""

    apply: Callable[[float, T, NamedArg(float, "error_power")], tuple[float, T]]
    r"""Propose a time-step $\Delta t$."""


def control_proportional_integral(
    *,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _Controller[float]:
    """Construct a proportional-integral-controller with time-clipping."""

    def init(_dt: float, /) -> float:
        return 1.0

    def apply(dt: float, error_power_prev: float, /, *, error_power):
        # Equivalent: error_power = error_norm ** (-1.0 / error_contraction_rate)
        a1 = error_power**power_integral_unscaled
        a2 = (error_power / error_power_prev) ** power_proportional_unscaled
        scale_factor_unclipped = safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt
        return dt_proposed, error_power_prev

    return _Controller(init=init, apply=apply)


def control_integral(
    *, safety=0.95, factor_min=0.2, factor_max=10.0
) -> _Controller[None]:
    """Construct an integral-controller."""

    def init(_dt, /) -> None:
        return None

    def apply(dt, _state, /, *, error_power):
        # error_power = error_norm ** (-1.0 / error_contraction_rate)
        scale_factor_unclipped = safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
        return scale_factor * dt, None

    return _Controller(init=init, apply=apply)
